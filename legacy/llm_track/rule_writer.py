"""rule_writer.py — Qwen writes game rules; the recordings check them (rule-finding, Stage A).

Runs in the SEPARATE .venv-llm (vLLM):

    make -C legacy/llm_track rules-smoke     # ft09, 2 rounds, training data only
    make -C legacy/llm_track rules           # ft09 + lp85 + ls20, 3 rounds, then the one test

What a rule is, how it is checked, the data split and the two baselines are all
in rule_referee.py (its RULE_DOC is exactly what Qwen reads).

THE LOOP
  Each game's training moves are split into ACTION GROUPS automatically: each
  button, and clicks by the colour clicked. For each group, Qwen sees the level's
  start board, where the ticker is, and 6 recorded moves of that group (before/after
  pictures with the changes boxed, plus the exact changed cells as text; moves
  that changed nothing are included when the group has them). It writes 6
  candidate rules. Nobody describes the game to it.
  Every candidate is checked on ALL of the game's training moves
  (rule_referee.check_rule). One that the sandbox rejects gets one repair
  attempt in the same round, and an answer that ran out of thinking room is
  asked for its code alone (added 2026-09-11 after a first Stage A attempt in
  which 41 of 138 round-1 answers reached no code - 23 of 24 on ls20 - while
  their thinking held sensible mechanics; kept in results/legacy_llm/rules/stageA_aborted_16k).
  Rounds 2-3: Qwen also sees the best rules so far for that group with their
  scores, moves where the best one was wrong (before | its prediction | the real
  after), and one move it did not cover. Then it writes 6 more.

  RULE BOOK: for each group, the accepted candidate that adds the most to
  "nothing changes" on the training moves (its GAIN: real changes it predicts
  exactly, minus moves where it predicts a change that did not happen; must be
  above 0). The book is ordered by training accuracy, then gain. A move is
  predicted by the first rule that applies, or "nothing changes" if none does.
  The book is picked on training moves only and scored ONCE on the test set.
  (Changed 2026-09-11 after the first ft09 smoke run, on training moves only: 36
  of 40 round-1 rules said "clicking colour X does nothing". Such a rule never
  changes a prediction, yet under the first selection rule, most exactly-right
  predictions, it would have won its group. The feedback now ranks by gain, and
  RULE_DOC says that "does nothing" rules add nothing.)

STAGE A - criteria fixed 2026-09-11, BEFORE any Qwen rule-writing run:
  R1 better          on at least 2 of the 3 games (ft09, lp85, ls20), the rule
     predictions     book's exact-prediction rate on the test set beats BOTH
                     baselines ('nothing' and 'memory', rule_referee.py), with the
                     95% CI of each difference (bootstrap over runs) above 0
  R2 rules exist     every game's rule book has at least one rule, i.e. an
                     accepted rule that predicts changes (tightened 2026-09-11,
                     before the Stage A run: "an accepted rule" alone was met by
                     "does nothing" rules)
  GO if R1 and R2. The model (MODEL below: the largest that fits the 5090) is
  chosen before the run.

Nothing from the test set (no board, no score) is ever put in a prompt.
"""
import argparse
import collections
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from .heur_writer import SAMPLING, extract_code, image_part
from .probe_images import BOX, GAP, GUTTER, PALETTE_NAMES, render_board, render_move
from .probe_images import CELL as IMG_CELL
from .rule_referee import (ACCEPT, DATA_DIR, GAMES, RULE_DOC, Act, RuleData, baseline_memory,
                           baseline_nothing, check_rule, compare, rule_book)
from .serializer import serialize
from .tickers import connected_components

# Chosen 2026-09-11, before the Stage A run, from ft09 smoke runs (training moves only,
# same pipeline and data): Qwen3.5-35B-A3B and the dense Qwen3.6-27B found the same
# tile-flip mechanic; the 35B, the largest model that runs on the 5090, took 12 min vs 43.
MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
# An overnight budget: 40 candidate rules per action group, with 4 rounds of
# counterexample feedback (was 3 x 6). Raised 2026-09-11 for the Stage A run.
ROUNDS, PER_GROUP = 5, 8
N_EXAMPLES, N_UNCHANGED = 6, 2     # moves shown per group; up to 2 of them unchanged
CELL = 6                           # px per cell: a board is 384x384 = 144 visual tokens
# Thinking is ON by default here (unlike heur_writer): in the 2026-09-11 smoke
# tests Qwen3.5-35B without thinking ran past 4,096 tokens without writing code on
# 5 of 6 toy answers, while with thinking it wrote ft09's tile rules.
THINKING = True
MAX_TOKENS = {False: 4096, True: 20480}
# 32768 is the most this fits: with 10 images allowed per prompt, vLLM's profiling
# leaves only 2.2 GiB of KV cache at --gpu-mem 0.90, and 36864 left 0.35 GiB and would
# not start (2026-09-11).
MAX_MODEL_LEN = {False: 16384, True: 32768}
AREAS_SHOWN = 24                   # changed areas listed per move (lp85 moves change up to ~20)
WORKERS = 8                        # rules checked at once, each in its own process

LEGEND = ", ".join(f"{i} {n}" for i, n in enumerate(PALETTE_NAMES))
SYSTEM = ("You work out the exact mechanics of an unknown grid game from recorded moves, "
          "and write each one down as a small Python rule.\n\n" + RULE_DOC +
          f"\n\nColour numbers are drawn as: {LEGEND}.")
TASK = ("Write ONE rule for this action group: what exactly does this action do to the board? "
        "Your rule is checked on ALL recorded moves of this game, not only the ones shown, so "
        "applies() must return False for moves it does not describe. It is also tested on moves "
        "you have not seen (another level, or other players' runs), so describe the mechanism, "
        "not these particular positions.")
ANSWER_FORMAT = {
    False: "First describe in at most 5 sentences exactly what this action does: which cells "
           "change, to which colours, and when nothing changes. Then give exactly one ```python "
           "code block that defines RULE, applies and predict.",
    True: "Think it through, but keep it brief: an answer that runs out of room before the "
          "code is wasted. Then give exactly one ```python code block that defines RULE, "
          "applies and predict.",
}
def _tail(text, limit=24000):
    """What the second try is shown of the first answer. A truncated answer can be
    20k tokens of thinking on its own, which would not fit beside a fresh reply."""
    answer = text.split("</think>")[-1].strip() if "</think>" in text else text
    return answer if len(answer) <= limit else "[...earlier reasoning omitted...]\n" + answer[-limit:]


REPAIR = ("Your rule could not be used: {reason}\nFix it and answer with exactly one ```python "
          "code block that defines RULE, applies and predict.")
FINISH = ("You ran out of room before writing the code. Write it now, with no further analysis: "
          "exactly one ```python code block that defines RULE, applies and predict.")


# ------------------------------------------------------------------ describing moves
def _areas(a, b, mask, label, limit):
    """Connected areas where boards a and b differ, with their colour flows."""
    comps = sorted(connected_components((a != b) & ~mask), key=len, reverse=True)
    out = []
    for c in comps[:limit]:
        ys, xs = c[:, 0], c[:, 1]
        flow = collections.Counter(zip(a[ys, xs].tolist(), b[ys, xs].tolist()))
        out.append(f"rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()} ({len(c)} cells): "
                   + ", ".join(label(p, q, n) for (p, q), n in flow.most_common(4)))
    if len(comps) > limit:
        out.append(f"... and {len(comps) - limit} more areas "
                   f"({sum(len(c) for c in comps[limit:])} cells)")
    return out


def move_text(data, i):
    """One recorded move, exactly: the action, what was clicked, and every change."""
    b, a, ai = data.before[i], data.after[i], int(data.actions[i])
    act = Act(ai)
    if act.click is None:
        head = f"ACTION{act.action}"
    else:
        y, x = act.click
        c = int(b[y, x])
        reg = next(r for r in connected_components(b == c) if ((r[:, 0] == y) & (r[:, 1] == x)).any())
        head = (f"click at (row {y}, col {x}) on colour {c} ({PALETTE_NAMES[c]}); the clicked "
                f"same-colour area is {len(reg)} cells, rows {reg[:, 0].min()}-{reg[:, 0].max()}, "
                f"cols {reg[:, 1].min()}-{reg[:, 1].max()}")
    moved = [f"a {m['size']}-cell colour-{m['color']} shape moved by ({m['dy']:+d} row, "
             f"{m['dx']:+d} col)" for m in serialize(b, ai, a, data.mask)["moves"]]
    changes = _areas(b, a, data.mask, lambda p, q, n: f"{p}->{q} x{n}", AREAS_SHOWN) or \
        ["nothing changed (outside the ticker)"]
    return head + "\n  " + "\n  ".join(moved + changes)


def render_compare(before, pred, actual, mask, cell=CELL):
    """BEFORE | the rule's prediction | the real AFTER, boxing where the last two differ."""
    side, gap = 64 * cell, GAP * cell // IMG_CELL
    img = Image.new("RGB", (3 * side + 2 * gap, side), GUTTER)
    for k, f in enumerate((before, pred, actual)):
        img.paste(Image.fromarray(render_board(f, cell)), (k * (side + gap), 0))
    draw = ImageDraw.Draw(img)
    for c in sorted(connected_components((pred != actual) & ~mask), key=len, reverse=True)[:8]:
        (y0, x0), (y1, x1) = c.min(axis=0), c.max(axis=0)
        for k in (1, 2):
            ox = k * (side + gap)
            draw.rectangle([ox + max(int(x0) * cell - 2, 0), max(int(y0) * cell - 2, 0),
                            ox + min((int(x1) + 1) * cell + 1, side - 1),
                            min((int(y1) + 1) * cell + 1, side - 1)], outline=BOX, width=2)
    return img


def ticker_line(mask):
    if not mask.any():
        return "This game has no ticker cells."
    ty, tx = np.nonzero(mask)
    return (f"Ticker cells (a timer/counter that changes by itself; ignored when rules are "
            f"checked): {int(mask.sum())} cells in rows {ty.min()}-{ty.max()}, cols "
            f"{tx.min()}-{tx.max()}.")


# ------------------------------------------------------------------ the prompts
def group_prompt(data, group, rng):
    """What round 1 shows Qwen for one action group - training moves only."""
    idx = data.idx("train", group)
    changed = np.array([not np.array_equal(data.before[i][data.live], data.after[i][data.live])
                        for i in idx])
    n_unch = min(N_UNCHANGED, int((~changed).sum()))
    pick = list(rng.choice(idx[changed], min(N_EXAMPLES - n_unch, int(changed.sum())), replace=False))
    pick += list(rng.choice(idx[~changed], n_unch, replace=False))
    rng.shuffle(pick)
    sizes = [int(((data.before[i] != data.after[i]) & data.live).sum()) for i in idx[changed]]
    stats = (f"The recordings have {len(idx)} training moves in this group; the board changed "
             f"(outside the ticker) on {changed.mean():.0%} of them"
             + (f", median {int(np.median(sizes))} cells changed." if sizes else "."))
    parts = [{"type": "text", "text": "The board at the start of a level of this game:"},
             image_part(render_board(data.firsts[data.first_idx[pick[0]]], CELL)),
             {"type": "text", "text":
              f"{ticker_line(data.mask)}\n\nACTION GROUP: {group}.\n{stats}\n\nRecorded moves "
              "of this group. Each picture shows the board BEFORE the move (left) and AFTER it "
              "(right); a cyan box marks what changed."}]
    for k, i in enumerate(pick, 1):
        parts += [{"type": "text", "text": f"Move {k}: {move_text(data, i)}"},
                  image_part(render_move(data.before[i], data.after[i], data.mask, cell=CELL))]
    return parts


def _rank(c):
    return (c["gain"], c["accepted"], c["accuracy"])


def feedback(cands, data, group, rng):
    """What rounds 2+ add for one group: the best rules so far and their mistakes."""
    mine = [c for c in cands if c["game"] == data.game and c["group"] == group]
    checked = sorted((c for c in mine if c["stage"] == "checked"), key=_rank, reverse=True)[:3]
    lines = ["Rules tried so far for this group, best first:"]
    for c in checked:
        lines += [f"\nRULE: {c['rule']}\nIt applies to {c['n_applies']} of the {c['n']} recorded "
                  f"moves ({c['in_group']} of this group's {c['group_size']}), is exactly right "
                  f"on {c['accuracy']:.0%} of them, and predicts {c['changes_right']} real changes "
                  "exactly" + (" - ACCEPTED." if c["accepted"] else f" - not accepted: {c['reason']}."),
                  "```python", c["code"], "```"]
    if not checked:
        lines.append("\n(none has run successfully yet)")
    parts = [{"type": "text", "text": "\n".join(lines)}]
    if checked:
        best = checked[0]
        for j in rng.permutation(len(best["_wrong"]))[:2]:
            i, pred = best["_wrong"][j]
            parts += [{"type": "text", "text":
                       f"The best rule was WRONG on this move: {move_text(data, i)}\nWhere its "
                       "prediction differs from the real board:\n  " + "\n  ".join(
                           _areas(pred, data.after[i], data.mask,
                                  lambda p, q, n: f"predicted {p}, really {q} x{n}", 12)) +
                       "\nPicture: BEFORE | THE RULE'S PREDICTION | REAL AFTER (cyan boxes: "
                       "where the prediction is wrong)."},
                      image_part(render_compare(data.before[i], pred, data.after[i], data.mask))]
        if best["_missed"]:
            i = int(rng.choice(best["_missed"]))
            parts += [{"type": "text", "text": "The best rule did not apply to this move of the "
                       f"group, where the board DID change: {move_text(data, i)}"},
                      image_part(render_move(data.before[i], data.after[i], data.mask, cell=CELL))]
    rej = collections.Counter(c["reason"][:90] for c in mine
                              if c["stage"] not in ("checked", "duplicate"))
    tail = ["\nSome candidates could not be used:"] + [f"  {n} x {why}" for why, n in rej.most_common(4)] \
        if rej else []
    tail.append("\nWrite ONE new rule for this group that is exactly right more often or covers "
                "more of its moves: improve the best one or try a different idea.")
    parts.append({"type": "text", "text": "\n".join(tail)})
    return parts


# ------------------------------------------------------------------ checking
def check_all(items, d):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(lambda c: check_rule(c["code"], c["game"], d, split="train"), items))


def record(c, r, data):
    c.update(stage=r["stage"], reason=r["reason"])
    if r["stage"] != "checked":
        return
    in_g = data.group[r["idx"]] == data.groups.index(c["group"])
    ch = data.changed[r["idx"]]
    right_ch = int((r["correct"] & ch).sum())
    c.update(rule=r["rule"], n=r["n"], n_applies=r["n_applies"], n_correct=r["n_correct"],
             accuracy=r["accuracy"], coverage=r["coverage"], accepted=r["accepted"],
             ms_mean=r["ms_mean"], errors=r["errors"], group_size=int(in_g.sum()),
             in_group=int((r["applies"] & in_g).sum()),
             in_group_correct=int((r["correct"] & in_g).sum()),
             # gain: what the rule adds to "nothing changes" (see RULE BOOK)
             changes_right=right_ch,
             gain=right_ch - int((r["applies"] & ~r["correct"] & ~ch).sum()),
             _wrong=r["wrong"], _missed=r["idx"][in_g & ~r["applies"] & ch].tolist())


def fresh(c, **kw):
    """A copy of a candidate without its check results."""
    keep = ("game", "group", "round", "k", "gen_tokens", "finished", "plan")
    return {**{k: c[k] for k in keep}, **kw}


# ------------------------------------------------------------------ the report
def write_report(path, a, thinking, datas, cands, books, rounds, test=None, train=None):
    L = ["# Rule-finding Stage A: can Qwen write rules that predict unseen moves?", "",
         f"Model {a.model}, thinking {'on' if thinking else 'off'}, {a.rounds} rounds x {a.n} "
         "candidates per action group. A rule is accepted if it applies to at least "
         f"{ACCEPT['min_applies']} training moves and is exactly right on at least "
         f"{ACCEPT['min_accuracy']:.0%} of them. Rule books are picked on training moves only; "
         "the test set is scored once.", "",
         "| round | generation | checking | candidates | checked | accepted | useful | other |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rounds:
        L.append(f"| {r['round']} | {r['gen_s']:.0f} s | {r['check_s']:.0f} s | {r['n']} | "
                 f"{r['stages'].get('checked', 0)} | {r['accepted']} | {r.get('useful', '-')} | "
                 f"{ {k: v for k, v in r['stages'].items() if k != 'checked'} } |")
    for g, data in datas.items():
        m = data.meta
        L += ["", f"## {g}", "",
              f"{m['split']} split: {m['n_train']} training moves in {len(data.groups)} groups, "
              f"{m['n_test']} test moves." + (
                  f" On its own training moves the rule book ({train[g]['n_rules']} rules) is "
                  f"exactly right on {train[g]['book']:.3f}, vs {train[g]['nothing']:.3f} for "
                  "'nothing changes'." if train else ""), "",
              "Best rule per group, by gain (changes predicted exactly, minus changes predicted "
              "where nothing happened):", "",
              "| action group | training moves | checked | accepted | best rule | applies | right "
              "| changes right | gain |",
              "|---|---|---|---|---|---|---|---|---|"]
        for grp in data.groups:
            cs = [c for c in cands if c["game"] == g and c["group"] == grp and c["stage"] == "checked"]
            best = max(cs, key=_rank) if cs else None
            L.append(f"| {grp} | {m['train_per_group'][grp]} | {len(cs)} | "
                     f"{sum(c['accepted'] for c in cs)} | "
                     + (f"{best['rule'][:90]} | {best['n_applies']} | {best['accuracy']:.1%} | "
                        f"{best['changes_right']} | {best['gain']} |" if best else "- | - | - | - | - |"))
    if test:
        L += ["", "## The test (criteria fixed before running; see rule_writer.py)", "",
              "Exact-prediction rate on the test moves, with 95% CIs bootstrapped over runs.", "",
              "| game | rule book | nothing | memory | book - nothing | book - memory | beats both |",
              "|---|---|---|---|---|---|---|"]
        for g, t in test["games"].items():
            r = t["rates"]
            ci = lambda n: f"{r[n]['rate']:.3f} ({r[n]['ci'][0]:.3f}-{r[n]['ci'][1]:.3f})"
            L.append(f"| {g} | {ci('rule book')} | {ci('nothing')} | {ci('memory')} | "
                     f"{r['nothing']['diff_vs_rule book'][0]:+.3f} to "
                     f"{r['nothing']['diff_vs_rule book'][1]:+.3f} | "
                     f"{r['memory']['diff_vs_rule book'][0]:+.3f} to "
                     f"{r['memory']['diff_vs_rule book'][1]:+.3f} | "
                     f"{'yes' if t['beats_both'] else 'no'} |")
        L += ["", f"- [{'PASS' if test['R1'] else 'FAIL'}] R1 better predictions: the rule book "
              f"beats both baselines on {test['n_beats']} of 3 games (needs 2)",
              f"- [{'PASS' if test['R2'] else 'FAIL'}] R2 rules exist: every game has at least one "
              "accepted rule", "", f"**{'GO' if test['R1'] and test['R2'] else 'NO-GO'}**"]
    L += ["", "## Rule books (test columns are descriptive only)", ""]
    for g, book in books.items():
        L += [f"### {g}", ""]
        if not book:
            L += ["(no accepted rules)", ""]
        for k, c in enumerate(book, 1):
            tt = (test or {}).get("games", {}).get(g, {}).get("per_rule", {}).get(k - 1)
            L.append(f"{k}. **{c['rule']}** (from {c['group']}, round {c['round']}): training "
                     f"applies {c['n_applies']}, right {c['accuracy']:.1%}, gain {c['gain']}"
                     + (f"; test applies {tt['n_applies']}, right {tt['accuracy']:.1%}" if tt else ""))
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--n", type=int, default=PER_GROUP, help="candidates per group per round")
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--out", default="results/legacy_llm/rules")
    ap.add_argument("--no-test", action="store_true", help="smoke run: never touch the test set")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=THINKING)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    thinking = a.thinking

    games = a.games.split(",")
    out = os.path.join(a.out, "stageA_smoke" if a.no_test else "stageA")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    datas = {g: RuleData(g, a.dir) for g in games}
    prompts = {(g, grp): group_prompt(data, grp, rng) for g, data in datas.items()
               for grp in data.groups}
    print(f"[rules] {len(prompts)} action groups over {', '.join(games)}", flush=True)

    from .judge import setup_vllm_env
    setup_vllm_env()
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, seed=a.seed, max_model_len=a.max_model_len or MAX_MODEL_LEN[thinking],
              gpu_memory_utilization=a.gpu_mem, max_num_seqs=32, max_num_batched_tokens=4096,
              limit_mm_per_prompt={"image": 1 + N_EXAMPLES + 3, "video": 0})
    print(f"[rules] {a.model} loaded in {time.time() - t0:.0f}s", flush=True)
    chat = dict(use_tqdm=False, chat_template_kwargs={"enable_thinking": thinking})

    cands, seen, rounds = [], set(), []
    raw_path = os.path.join(out, "raw_outputs.jsonl")
    open(raw_path, "w").close()
    for rnd in range(1, a.rounds + 1):
        keys = list(prompts)
        convs = []
        for g, grp in keys:
            user = list(prompts[(g, grp)])
            if rnd > 1:
                user += feedback(cands, datas[g], grp, rng)
            user.append({"type": "text", "text": TASK + "\n\n" + ANSWER_FORMAT[thinking]})
            convs.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        sp = SamplingParams(n=a.n, max_tokens=MAX_TOKENS[thinking], seed=a.seed * 100 + rnd,
                            **SAMPLING[thinking])
        tg = time.time()
        outs = llm.chat(convs, sp, **chat)
        gen_s = time.time() - tg

        new, retry, n_round = [], [], len(cands)
        with open(raw_path, "a") as f:
            for (g, grp), conv, req in zip(keys, convs, outs):
                for k, o in enumerate(req.outputs):
                    f.write(json.dumps({"round": rnd, "game": g, "group": grp, "k": k,
                                        "text": o.text, "finish": o.finish_reason}) + "\n")
                    code = extract_code(o.text)
                    c = {"game": g, "group": grp, "round": rnd, "k": k, "code": code,
                         "gen_tokens": len(o.token_ids), "finished": o.finish_reason == "stop",
                         "plan": o.text.split("</think>")[-1].split("```")[0].strip()[:600]}
                    h = hashlib.sha1(f"{g}\n{code}".encode()).hexdigest()
                    if code is None:
                        c.update(stage="no-code", reason="no ```python block in the answer"
                                 + ("" if c["finished"] else " (ran out of tokens)"))
                        if not c["finished"]:          # ask for the code on its own
                            retry.append((c, conv, o.text))
                    elif h in seen:
                        c.update(stage="duplicate", reason="same code as an earlier candidate")
                    else:
                        seen.add(h)
                        new.append((c, conv, o.text))
                    cands.append(c)
        tc = time.time()
        for (c, _, _), r in zip(new, check_all([c for c, _, _ in new], a.dir)):
            record(c, r, datas[c["game"]])

        # ---- one repair attempt for code the sandbox rejected
        broken = [(c, conv, text) for c, conv, text in new
                  if c["stage"] in ("static", "runtime", "timeout")] + retry
        if broken:
            # the second try never thinks again: the model has already reasoned, and for a
            # truncated answer more thinking is exactly what ran out of room
            rconvs = [conv + [{"role": "assistant", "content": _tail(text)},
                              {"role": "user", "content": FINISH if c["stage"] == "no-code"
                               else REPAIR.format(reason=c["reason"])}]
                      for c, conv, text in broken]
            rsp = SamplingParams(n=1, max_tokens=MAX_TOKENS[False], seed=a.seed * 100 + rnd + 50,
                                 **SAMPLING[False])
            fixed = []
            for (c, _, _), req in zip(broken, llm.chat(rconvs, rsp, use_tqdm=False,
                                                       chat_template_kwargs={"enable_thinking": False})):
                code = extract_code(req.outputs[0].text)
                h = hashlib.sha1(f"{c['game']}\n{code}".encode()).hexdigest()
                if code is not None and h not in seen:
                    seen.add(h)
                    fixed.append(fresh(c, code=code, repaired=True))
            for c2, r in zip(fixed, check_all(fixed, a.dir)):
                record(c2, r, datas[c2["game"]])
                cands.append(c2)
        this = cands[n_round:]
        rounds.append({"round": rnd, "gen_s": gen_s, "check_s": time.time() - tc, "n": len(this),
                       "stages": dict(collections.Counter(c["stage"] for c in this)),
                       "accepted": sum(c.get("accepted", False) for c in this),
                       "useful": sum(c.get("accepted", False) and c["gain"] > 0 for c in this)})
        print(f"[rules] round {rnd}: {rounds[-1]['stages']}, {rounds[-1]['accepted']} accepted, "
              f"{rounds[-1]['useful']} of them predict changes; {gen_s:.0f}s generation, "
              f"{rounds[-1]['check_s']:.0f}s checking", flush=True)
        for g in games:
            use = [c for c in cands if c["game"] == g and c.get("accepted") and c["gain"] > 0]
            print(f"    {g}: {len({c['group'] for c in use})} of {len(datas[g].groups)} groups "
                  f"have an accepted rule that predicts changes", flush=True)

    # ---- rule books, from training moves only
    books = {}
    for g, data in datas.items():
        best = []
        for grp in data.groups:
            acc = [c for c in cands if c["game"] == g and c["group"] == grp
                   and c.get("accepted") and c["gain"] > 0]
            if acc:
                best.append(max(acc, key=lambda c: (c["gain"], c["accuracy"])))
        books[g] = sorted(best, key=lambda c: (-c["accuracy"], -c["gain"]))
        with open(os.path.join(out, f"rulebook_{g}.py"), "w") as f:
            f.write(f"# Rule book for {g}: the first rule that applies predicts the move.\n")
            for k, c in enumerate(books[g], 1):
                f.write(f"\n# ---- rule {k}: from '{c['group']}', round {c['round']}; training: "
                        f"applies {c['n_applies']}, right {c['accuracy']:.1%}, gain {c['gain']}\n"
                        f"{c['code']}\n")
    with open(os.path.join(out, "candidates.jsonl"), "w") as f:
        for c in cands:
            f.write(json.dumps({k: v for k, v in c.items() if not k.startswith("_")}) + "\n")

    train = {}                              # descriptive: each book on its own training moves
    for g, data in datas.items():
        tr = data.idx("train")
        nothing = rate = float((~data.changed[tr]).mean())
        if books[g]:
            bk = rule_book([c["code"] for c in books[g]], g, a.dir, split="train")
            rate = float(bk["correct"].mean()) if bk["ok"] else float("nan")
        train[g] = {"book": round(rate, 4), "nothing": round(nothing, 4), "n_rules": len(books[g])}
        print(f"[rules] {g}: rule book of {len(books[g])} rules is exactly right on {rate:.3f} of "
              f"its training moves ('nothing changes': {nothing:.3f})", flush=True)

    test = None
    if not a.no_test:                       # ---- the one look at the test set
        test = {"games": {}}
        for g, data in datas.items():
            te, tr = data.idx("test"), data.idx("train")
            nothing = baseline_nothing(data, te)
            if books[g]:
                bk = rule_book([c["code"] for c in books[g]], g, a.dir)
                assert bk["ok"] and np.array_equal(bk["idx"], te), bk.get("reason")
                book_correct = bk["correct"]
            else:
                book_correct = nothing
            rates, n_runs = compare(data, te, {"rule book": book_correct, "nothing": nothing,
                                                "memory": baseline_memory(data, tr, te)})
            beats = all(rates[n]["diff_vs_rule book"][0] > 0 for n in ("nothing", "memory"))
            per_rule = {}
            for k, c in enumerate(books[g]):             # descriptive only
                r = check_rule(c["code"], g, a.dir, split="test")
                if r["stage"] == "checked":
                    per_rule[k] = {"n_applies": r["n_applies"], "accuracy": r["accuracy"]}
            test["games"][g] = {"rates": rates, "n_test_runs": n_runs, "beats_both": beats,
                                "per_rule": per_rule}
        test["n_beats"] = sum(t["beats_both"] for t in test["games"].values())
        test["R1"] = test["n_beats"] >= 2
        test["R2"] = all(len(books[g]) > 0 for g in datas)
        with open(os.path.join(out, "test.json"), "w") as f:
            json.dump(test, f, indent=1)

    write_report(os.path.join(out, "report.md"), a, thinking, datas, cands, books, rounds, test, train)
    print(f"[rules] done in {time.time() - t0:.0f}s -> {out}/report.md", flush=True)


if __name__ == "__main__":
    main()
