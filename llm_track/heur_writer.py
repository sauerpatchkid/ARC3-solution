"""heur_writer.py — Qwen writes heuristics; the recordings decide (Probe C).

Runs in the SEPARATE .venv-llm (vLLM):

    make -C llm_track probe-c
    HF_HUB_OFFLINE=1 .venv-llm/bin/python -m llm_track.heur_writer --game ft09

THE LOOP (heuristics plan, step 2)
  Round 1: Qwen sees LEVEL 1 of the game - the start board, the board one move
  before it was solved, the layout, what changed between them, and a few moves
  that helped or hurt - plus the helper API (heur_api.API_DOC). Asked for a
  short plan and then the code, and sampled for variety, it writes N candidate
  heuristics. Each goes
  through the sandbox and is graded by the referee on LEVEL 1 only.
  Rounds 2..R: Qwen sees the best candidates so far with their level-1 scores,
  the moves the best one got most wrong, and why rejected ones failed, and
  writes N more.

  REPAIR: a candidate rejected by the sandbox or its gates gets one repair
  attempt within its round - Qwen sees its own answer and the rejection reason
  and returns fixed code. Standard for LLM code-writing loops; added after the
  level-1 smoke run, where 3 of 4 candidates failed on API mistakes.

  The WINNER is the best level-1 score over all rounds. Only then is it graded
  on LEVEL 2 - a level Qwen never saw and the loop never scored against. That
  single number decides Probe C. Every graded candidate is also scored on
  level 2 for a descriptive table, which plays no part in the decision.

PROBE C - criteria fixed 2026-09-11, BEFORE any Qwen run:
  C1 signal        the winner's level-2 AUC has a 95% CI lower bound above 0.50
  C2 beyond        the winner's level-2 AUC is above the best simple rule's
     simple rules  level-2 AUC on the same referee (heur_referee.SIMPLE_RULES)
  C3 viable        in at least 3 of the 4 rounds, at least one candidate passes
                   the sandbox and its gates
  GO if all three pass. Primary game: ft09, the only game with enough level-2
  completions when this was written. Any other game is descriptive only.

Nothing about level 2 - no board, no label, no score - is ever put in a prompt.
"""
import argparse
import base64
import collections
import hashlib
import io
import json
import os
import re
import time

import numpy as np
from PIL import Image

from .corpus import CorpusReader
from .heur_api import API_DOC, regions, background_of
from .heur_referee import Referee, simple_rule_scores
from .heur_sandbox import evaluate
from .probe_images import PALETTE_NAMES, render_board, render_move
from .probe_pairs import segment_start
from .serializer import serialize
from .tickers import connected_components

TRAIN_LEVEL, TEST_LEVEL = 0, 1         # level 1 to learn from, level 2 to test on
ROUNDS, PER_ROUND = 4, 8
CELL = 6               # px per cell in prompt pictures: a board is 384x384 = 144 visual tokens
N_EXAMPLES = 2         # helped moves and hurt moves shown, each
# Thinking is OFF by default. Smoke run 2026-09-11 (level 1 only, never level 2):
# with thinking on, Qwen3.5-9B spent all 8,192 tokens trying to work out the
# puzzle's rule and never reached code (2 of 2 candidates). So the answer format
# is "a short plan, then the code" - the reason-first idea that fixed Probe A's
# prompt v2 - and the search comes from sampling many candidates and grading
# them, not from one long deliberation. --thinking restores it, with more room.
MAX_TOKENS = {False: 4096, True: 16384}
MAX_MODEL_LEN = {False: 12288, True: 24576}
SAMPLING = {False: dict(temperature=0.7, top_p=0.8, top_k=20),    # Qwen's non-thinking defaults
            True: dict(temperature=0.6, top_p=0.95, top_k=20)}    # Qwen's thinking defaults

SYSTEM = ("You write small, fast Python scoring functions (\"heuristics\") that help "
          "an agent play an unknown grid puzzle game. The agent does not know the rules "
          "or the goal; it explores by clicking cells or pressing buttons. Your heuristic "
          "tells it which actions look most promising on the current board.\n\n" + API_DOC)

LEGEND = ", ".join(f"{i} {n}" for i, n in enumerate(PALETTE_NAMES))

TASK = """Write a heuristic for this game. It will be used on OTHER levels of this game, \
whose solved boards you will not see. So capture the RULE that makes a move helpful - \
what the player should aim for, and which actions get there - not the specific \
positions in this level."""

ANSWER_FORMAT = {
    False: "First write a short plan - at most 6 sentences: what you think the goal is, and "
           "which actions move the board toward it. Then give exactly one ```python code "
           "block that defines IDEA and score(board, api).",
    True: "Think it through, then answer with exactly one ```python code block that defines "
          "IDEA and score(board, api).",
}


# ------------------------------------------------------------------ helpers
def png_url(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def image_part(img):
    return {"type": "image_url", "image_url": {"url": png_url(img)}}


def extract_code(text):
    """The last ```python block of the final answer (after any thinking)."""
    answer = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", answer, re.S)
    return blocks[-1].strip() if blocks else None


class Transitions:
    """Finds a referee sample's full transition (with its next board) in the corpus."""

    def __init__(self, ref):
        self.ref, self.readers = ref, {}

    def get(self, i):
        ev = self.ref.events[int(self.ref.event[i])]
        d = ev["corpus"]
        if d not in self.readers:
            r = CorpusReader(d)
            self.readers[d] = (r, {int(n): k for k, n in enumerate(r.scalars["action_nums"])})
        r, idx = self.readers[d]
        b = idx[int(ev["event"].rsplit("/", 1)[1])]
        j = segment_start(r.scalars["levels"], b) + int(self.ref.moves_in_level[i])
        t = r.get(j)
        assert (t["frame"] == self.ref.boards[i]).all(), "referee sample / corpus mismatch"
        return t


def describe(ref, trans, i):
    t = trans.get(i)
    g = ref.events[int(ref.event[i])]["game"]
    return t, serialize(t["frame"], t["action"], t["next_frame"], ref.masks[g], game=g)["text"]


# ------------------------------------------------------------------ the prompt
def build_context(ref, trans, game, rng):
    """Everything round 1 shows Qwen, all from ONE level-1 completion."""
    train = ref.select(game=game, level=TRAIN_LEVEL)
    events = sorted({int(ref.event[i]) for i in train},
                    key=lambda e: -ref.events[e]["sampled"])
    e = events[0]                                   # the most-sampled level-1 completion
    here = train[ref.event[train] == e]
    tick = ref.masks[game]
    start, solved = ref.first[e], ref.s_star[e]
    ty, tx = np.nonzero(tick)
    tick_line = (f"Ticker cells (a timer/counter that changes by itself - ignore it): "
                 f"{int(tick.sum())} cells in rows {ty.min()}-{ty.max()}, cols "
                 f"{tx.min()}-{tx.max()}." if tick.any() else "This game has no ticker cells.")
    lay = regions(start, background_of(start), tick)

    lay_lines = [f"  colour {r.colour} ({PALETTE_NAMES[r.colour]}), {r.size} cells, "
                 f"rows {r.y0}-{r.y1}, cols {r.x0}-{r.x1}" for r in lay[:40]]
    if len(lay) > 40:
        lay_lines.append(f"  ... and {len(lay) - 40} smaller regions")
    diff = sorted(connected_components((start != solved) & ~tick), key=len, reverse=True)
    diff_lines = []
    for c in diff[:20]:
        ys, xs = c[:, 0], c[:, 1]
        b = int(np.bincount(start[ys, xs], minlength=16).argmax())
        a = int(np.bincount(solved[ys, xs], minlength=16).argmax())
        diff_lines.append(f"  rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()} "
                          f"({len(c)} cells): colour {b} ({PALETTE_NAMES[b]}) -> "
                          f"{a} ({PALETTE_NAMES[a]})")

    examples = []
    for kind, lab in (("HELPED", 1), ("HURT", -1)):
        pool = here[ref.labels[here] == lab]
        for i in rng.choice(pool, min(N_EXAMPLES, len(pool)), replace=False):
            t, text = describe(ref, trans, int(i))
            examples.append({"kind": kind, "text": text,
                             "image": render_move(t["frame"], t["next_frame"], tick, cell=CELL)})
    rng.shuffle(examples)

    parts = [{"type": "text", "text": "Here is level 1 of the game. First picture: the "
              "board at the START of the level. Second picture: the board one move before "
              "the level was SOLVED."},
             image_part(render_board(start, CELL)), image_part(render_board(solved, CELL)),
             {"type": "text", "text":
              f"Colour numbers are drawn as: {LEGEND}.\n\n{tick_line}\n\n"
              f"Regions on the start board, largest first:\n" + "\n".join(lay_lines) +
              "\n\nWhat differs between the start board and the solved board:\n" +
              ("\n".join(diff_lines) or "  (nothing)") +
              "\n\nMoves a player made in this level. Each picture shows the board BEFORE "
              "the move (left) and AFTER it (right); a cyan box marks what changed. HELPED "
              "means the move brought the board closer to the solved board; HURT means it "
              "moved the board further away."}]
    for k, ex in enumerate(examples, 1):
        parts += [{"type": "text", "text": f"Example {k} ({ex['kind']}): {ex['text']}"},
                  image_part(ex["image"])]
    return {"event": ref.events[e]["event"], "parts": parts,
            "n_images": 2 + len(examples)}


def feedback(cands, ref, trans):
    """What rounds 2+ add: the best so far, their mistakes, and the rejections."""
    graded = sorted((c for c in cands if c["stage"] == "graded"),
                    key=lambda c: -c["auc_train"])[:3]
    lines = ["Heuristics tried so far on level 1, best first (AUC: 0.5 = no better "
             "than chance, 1.0 = perfect):"]
    for c in graded:
        lines += [f"\nIDEA: {c['idea']}\nlevel-1 AUC: {c['auc_train']:.3f}",
                  "```python", c["code"], "```"]
    if not graded:
        lines.append("\n(none has passed the checks yet)")
    if graded and graded[0].get("_pct") is not None:
        pct, idx = graded[0]["_pct"], graded[0]["_idx"]
        lab = ref.labels[idx]
        lines.append("\nMoves the best heuristic got most wrong:")
        for i in np.argsort(-np.where(lab < 0, pct, -1))[:2]:          # hurt, ranked high
            lines.append(f"  It ranked this HURT move in its top {100 * (1 - pct[i]):.0f}%: "
                         f"{describe(ref, trans, int(idx[i]))[1]}")
        for i in np.argsort(np.where(lab > 0, pct, 2))[:2]:            # helped, ranked low
            lines.append(f"  It ranked this HELPED move in its bottom {100 * pct[i]:.0f}%: "
                         f"{describe(ref, trans, int(idx[i]))[1]}")
    rej = collections.Counter(c["reason"][:90] for c in cands if c["stage"] != "graded")
    if rej:
        lines.append("\nSome candidates were rejected:")
        lines += [f"  {n} x {why}" for why, n in rej.most_common(5)]
    lines.append("\nWrite ONE new heuristic that scores higher: improve one of these or try a "
                 "different idea.")
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="ft09")
    ap.add_argument("--referee", default="results/llm/referee/v2")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--n", type=int, default=PER_ROUND, help="candidates per round")
    ap.add_argument("--out", default="results/llm/probeC")
    ap.add_argument("--no-test", action="store_true",
                    help="smoke run: never touch level 2")
    ap.add_argument("--thinking", action="store_true",
                    help="thinking mode (off by default - see MAX_TOKENS)")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = os.path.join(a.out, a.game + ("_smoke" if a.no_test else ""))
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ref = Referee(a.referee)
    trans = Transitions(ref)
    train = dict(game=a.game, level=TRAIN_LEVEL)
    test = dict(game=a.game, level=TEST_LEVEL)
    n_train_ev = len({int(e) for e in ref.event[ref.select(**train)]})
    n_test_ev = len({int(e) for e in ref.event[ref.select(**test)]})
    print(f"[writer] {a.game}: {n_train_ev} level-1 completions to learn from, "
          f"{n_test_ev} level-2 completions to test on")
    ctx = build_context(ref, trans, a.game, rng)

    from .judge import setup_vllm_env
    setup_vllm_env()
    from vllm import LLM, SamplingParams
    import vllm
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", seed=a.seed,
              max_model_len=a.max_model_len or MAX_MODEL_LEN[a.thinking],
              gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.n,
              max_num_batched_tokens=4096,
              limit_mm_per_prompt={"image": ctx["n_images"], "video": 0})
    print(f"[writer] {a.model} loaded in {time.time() - t0:.0f}s")

    cands, seen, round_stats = [], set(), []
    raw_path = os.path.join(out, "raw_outputs.jsonl")
    open(raw_path, "w").close()
    for rnd in range(1, a.rounds + 1):
        user = list(ctx["parts"])
        body = TASK if rnd == 1 else TASK + "\n\n" + feedback(cands, ref, trans)
        user.append({"type": "text", "text": body + "\n\n" + ANSWER_FORMAT[a.thinking]})
        conv = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        sp = SamplingParams(n=a.n, max_tokens=MAX_TOKENS[a.thinking],
                            seed=a.seed * 100 + rnd, **SAMPLING[a.thinking])
        tg = time.time()
        outs = llm.chat([conv], sp, use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": a.thinking})[0].outputs
        gen_s = time.time() - tg
        with open(raw_path, "a") as f:
            for k, o in enumerate(outs):
                f.write(json.dumps({"round": rnd, "k": k, "text": o.text,
                                    "finish": o.finish_reason}) + "\n")

        new = 0
        answers = []
        for k, o in enumerate(outs):
            code = extract_code(o.text)
            c = {"round": rnd, "k": k, "code": code, "idea": "", "gen_tokens": len(o.token_ids),
                 "finished": o.finish_reason == "stop",
                 "plan": o.text.split("</think>")[-1].split("```")[0].strip()[:800]}
            if code is None:
                c.update(stage="no-code", reason="no ```python block in the answer"
                         + ("" if c["finished"] else " (ran out of tokens while thinking)"))
            else:
                h = hashlib.sha1(code.encode()).hexdigest()
                if h in seen:
                    c.update(stage="duplicate", reason="same code as an earlier candidate")
                else:
                    seen.add(h)
                    r = evaluate(code, a.referee, where=train, return_samples=True)
                    c.update(stage=r["stage"], reason=r["reason"], idea=r.get("idea", ""),
                             auc_train=r.get("auc"), ci_train=r.get("ci"),
                             ms_mean=r.get("ms_mean"), constant_frac=r.get("constant_frac"),
                             _idx=r.get("idx"), _pct=r.get("pct"))
                    new += r["stage"] == "graded"
            cands.append(c)
            answers.append(o.text)

        # ---- one repair attempt for code the sandbox or its gates rejected
        broken = [(c, t) for c, t in zip(cands[-len(outs):], answers)
                  if c["stage"] in ("static", "runtime", "gate", "timeout")]
        if broken:
            fix_convs = [conv + [{"role": "assistant", "content": t},
                                 {"role": "user", "content":
                                  f"That code was rejected: {c['reason']}\nFix it. Reply with "
                                  f"exactly one ```python code block defining IDEA and "
                                  f"score(board, api)."}] for c, t in broken]
            fsp = SamplingParams(n=1, max_tokens=MAX_TOKENS[a.thinking],
                                 seed=a.seed * 100 + rnd + 50, **SAMPLING[a.thinking])
            fixes = llm.chat(fix_convs, fsp, use_tqdm=False,
                             chat_template_kwargs={"enable_thinking": a.thinking})
            for (c, _), fo in zip(broken, fixes):
                code = extract_code(fo.outputs[0].text)
                c["first_try"] = {"stage": c["stage"], "reason": c["reason"]}
                if code is None:
                    c.update(stage="no-code", reason="repair returned no code")
                    continue
                h = hashlib.sha1(code.encode()).hexdigest()
                if h in seen:
                    c.update(stage="duplicate", reason="repair repeated earlier code")
                    continue
                seen.add(h)
                r = evaluate(code, a.referee, where=train, return_samples=True)
                c.update(code=code, stage=r["stage"], reason=r["reason"],
                         idea=r.get("idea", ""), auc_train=r.get("auc"), ci_train=r.get("ci"),
                         ms_mean=r.get("ms_mean"), constant_frac=r.get("constant_frac"),
                         _idx=r.get("idx"), _pct=r.get("pct"))
                new += r["stage"] == "graded"
            with open(raw_path, "a") as f:
                for (c, _), fo in zip(broken, fixes):
                    f.write(json.dumps({"round": rnd, "k": c["k"], "repair": True,
                                        "text": fo.outputs[0].text}) + "\n")
        best = max((c["auc_train"] for c in cands if c["stage"] == "graded"), default=None)
        st = collections.Counter(c["stage"] for c in cands if c["round"] == rnd)
        round_stats.append({"round": rnd, "gen_s": round(gen_s), "passed": new,
                            "repaired_ok": sum(1 for c in cands if c["round"] == rnd and
                                               "first_try" in c and c["stage"] == "graded"),
                            **st})
        print(f"[writer] round {rnd}: {dict(st)} in {gen_s:.0f}s generation; best level-1 "
              f"AUC so far {best if best is None else round(best, 3)}")

    # ---- the decision: winner picked on level 1, graded once on level 2
    graded = [c for c in cands if c["stage"] == "graded"]
    winner = max(graded, key=lambda c: (c["auc_train"], -c["ms_mean"])) if graded else None
    result = {"game": a.game, "model": a.model, "vllm": vllm.__version__,
              "thinking": a.thinking,
              "referee": a.referee, "example_event": ctx["event"],
              "rounds": round_stats, "n_candidates": len(cands), "n_graded": len(graded)}
    if winner and not a.no_test:
        wt = evaluate(winner["code"], a.referee, where=test)
        simple = {k: v for k, v in simple_rule_scores(a.referee, test).items() if v["ok"]}
        best_simple = max(simple.items(), key=lambda kv: kv[1]["auc"]) if simple else None
        for c in graded:                                    # descriptive only
            r = evaluate(c["code"], a.referee, where=test)
            c["auc_test"] = r.get("auc")
        passes = sum(1 for s in round_stats if s["passed"] > 0)
        checks = {
            "C1 signal: level-2 CI lower bound > 0.50": wt.get("ci", [0])[0] > 0.50,
            "C2 beyond simple rules: level-2 AUC > best simple rule": bool(
                best_simple and wt.get("auc", 0) > best_simple[1]["auc"]),
            f"C3 viable: a candidate passed in >= 3 of {a.rounds} rounds": passes >= 3,
        }
        result.update(winner={k: winner[k] for k in ("round", "k", "idea", "plan",
                                                     "auc_train", "ci_train", "ms_mean")},
                      winner_test=wt, simple_rules_test={k: v["auc"] for k, v in simple.items()},
                      checks=checks, go=all(checks.values()))
    elif winner:
        result["winner"] = {k: winner[k] for k in ("round", "k", "idea", "auc_train", "ms_mean")}

    with open(os.path.join(out, "candidates.jsonl"), "w") as f:
        for c in cands:
            f.write(json.dumps({k: v for k, v in c.items() if not k.startswith("_")}) + "\n")
    if winner:
        with open(os.path.join(out, "winner.py"), "w") as f:
            f.write(winner["code"] + "\n")
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    write_report(out, result, cands, a)
    print(f"[writer] done in {time.time() - t0:.0f}s -> {out}/report.md")


def write_report(out, res, cands, a):
    L = [f"# Probe C — can Qwen write a heuristic that works on an unseen level? ({res['game']})\n",
         f"Model {res['model']}, referee `{res['referee']}`, {a.rounds} rounds x {a.n} "
         f"candidates. Qwen was shown level 1 (completion `{res['example_event']}`); the "
         f"winner is picked on level 1 and graded once on level 2.\n",
         "| round | generation | graded | rejected / other |", "|---|---|---|---|"]
    for s in res["rounds"]:
        other = {k: v for k, v in s.items() if k not in ("round", "gen_s", "passed", "graded")}
        L.append(f"| {s['round']} | {s['gen_s']} s | {s.get('graded', 0)} | {other or '-'} |")
    if "checks" in res:
        w, wt = res["winner"], res["winner_test"]
        L += [f"\n## Decision (criteria fixed before running — see heur_writer.py)\n",
              f"Winner: round {w['round']}, level-1 AUC {w['auc_train']:.3f}, "
              f"{w['ms_mean']} ms/call. IDEA: _{w['idea']}_\n",
              f"Qwen's plan: _{w.get('plan', '').replace(chr(10), ' ')}_\n",
              f"**Level 2 (never seen): AUC {wt.get('auc', float('nan')):.3f}, "
              f"95% CI {wt.get('ci', ['-', '-'])[0]}–{wt.get('ci', ['-', '-'])[1]}**\n",
              "Simple rules on level 2: " + ", ".join(f"{k} {v:.3f}" for k, v in
                                                     res["simple_rules_test"].items()) + "\n"]
        L += [f"- [{'PASS' if ok else 'FAIL'}] {k}" for k, ok in res["checks"].items()]
        L.append(f"\n**{'GO' if res['go'] else 'NO-GO'}**\n")
        L += ["## All graded candidates (level-2 column is descriptive only)\n",
              "| round | level-1 AUC | level-2 AUC | ms/call | idea |", "|---|---|---|---|---|"]
        for c in sorted((c for c in cands if c["stage"] == "graded"),
                        key=lambda c: -c["auc_train"]):
            L.append(f"| {c['round']} | {c['auc_train']:.3f} | "
                     f"{c.get('auc_test') if c.get('auc_test') is None else round(c['auc_test'], 3)}"
                     f" | {c['ms_mean']} | {c['idea'][:90]} |")
    rej = collections.Counter(c["reason"][:80] for c in cands if c["stage"] != "graded")
    if rej:
        L += ["\n## Why candidates were rejected\n"] + [f"- {n} x {r}" for r, n in rej.most_common(8)]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
