#!/usr/bin/env python3
"""wm_refine.py - Rulebook v2, CP0: the near-miss refinement day (docs/plans/
rulebook-v2.md section 3.2; pre-registered in docs/plans/rulebook-v2-prereg.md
section 3 BEFORE any answer was generated).

Tier 0a left seven action groups whose best rule was 90-97% right and never
exact, and its settings never gave an admitted rule a second try. This is v1's
one allowed day of other settings, spent on exactly that: for each group, keep
refining the most promising rule, showing the model the cases it gets wrong,
until one is exact or the budget runs out.

THE BANDIT (REx, Tang et al. 2024). Every rule for the group is an arm. At each
step draw, for every arm,
        theta ~ Beta(1 + C*h, 1 + C*(1 - h) + N)
with h the arm's accuracy (exactly-right share of the recorded moves it applies
to), N how many times it has already been refined and C = 20, and refine the arm
with the highest draw: one LLM call that sees the group as Tier 0a showed it
(3 recorded moves instead of 6, to leave room), the rule, its scores, and up to
6 of its failing cases - wrong predictions first, then moves of the group it did
not cover. The child is checked on every case of the level, as in Tier 0a, and
joins the pool if it applies to >= 20 moves with gain > 0. A group stops at its
first EXACT rule or after 12 calls.

EXACT = v1's "plan-eligible", unchanged: admitted, right on every case it
applies to, and none of those cases has conflicting outcomes.

No new evidence: the frozen Tier 0a evidence files and candidates are the input.
Same model and settings as Tier 0a (thinking on, 20,480-token answers, one
code-only retry, one repair attempt), one candidate per call.

  experiments/rulebook/llm_run.sh start nearmiss          # server + this, detached
  uv run python tools/wm_refine.py --dry-run               # the starting pools, no LLM
Writes <out>/{<game>__<group>.json, report.md, llm_cache.jsonl}.
"""
import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
from wm import rules as R  # noqa: E402
from wm.check import ADMIT, check_rule, grade  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.llm import ChatClient  # noqa: E402
from wm.metrics import claimed_grade  # noqa: E402

V1 = os.path.join(ROOT, "results", "rulebook", "tier0a")
GROUPS = [("tu93", "ACTION1"), ("tu93", "ACTION3"),
          ("m0r0", "ACTION1"), ("m0r0", "ACTION2"), ("m0r0", "ACTION3"), ("m0r0", "ACTION4"),
          ("dc22", "ACTION4")]
CALLS, C = 12, 20.0              # refinement calls per group; REx's constant
N_SHOWN, N_FAILING = 3, 6        # recorded moves and failing cases in a refinement prompt
WORKERS = 4                      # groups refined at the same time
INSTRUCTION = (
    "Write ONE improved rule for this group. It must be exactly right on every move it covers, "
    "including the ones shown above: keep what the rule gets right and fix what it gets wrong. "
    "If a case cannot be predicted from the board alone, make applies() return False for it "
    "rather than guess.")


def rng_for(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def slug(group):
    return "".join(ch if ch.isalnum() else "_" for ch in group)


def evaluate(code, ev, ev_path, group):
    """Check one rule on the level; returns the arm record (stage, grade, ...)."""
    r = check_rule(code, ev_path)
    arm = {"code": code, "stage": r["stage"], "reason": r.get("reason", "ok")}
    if r["stage"] == "checked":
        g = grade(r, ev, group)
        in_g = np.array([x == group for x in ev.group])
        ch = in_g & ev.changed & ~ev.frozen
        arm.update(res=r, rule=r["rule"], grade=g, reason=g["reason"], v2=claimed_grade(r, ev),
                   group_changing_covered=round(float((r["applies"] & r["correct"] & ch).sum()) / max(int(ch.sum()), 1), 4))
    return arm


def in_pool(arm):
    return (arm["stage"] == "checked" and arm["grade"]["moves"] >= ADMIT["min_moves"]
            and arm["grade"]["gain"] > 0)


def is_exact(arm):
    return arm["stage"] == "checked" and arm["grade"]["plan_eligible"]


def exact_up_to_conflicts(arm):
    """Right on every case's usual outcome, but some of those cases have been
    seen to end differently: as exact as a rule over (screen, action) can be."""
    g = arm.get("grade")
    return bool(g and g["admitted"] and g["exact_all"] and g["conflict_keys"] > 0)


def starting_pool(game, group, ev, ev_path):
    """Tier 0a's candidates for the group, re-checked on the frozen evidence."""
    with open(os.path.join(V1, f"{game}.json")) as f:
        cands = json.load(f)["candidates"]
    codes = []
    for c in cands:
        code = (c.get("code") or "").strip()
        if c["group"] == group and code and code not in codes:
            codes.append(code)
    with ThreadPoolExecutor(8) as ex:
        arms = list(ex.map(lambda code: evaluate(code, ev, ev_path, group), codes))
    pool = [dict(a, refined=0, origin="tier0a", step=0) for a in arms if in_pool(a)]
    return pool, len(codes)


def pick(pool, rng):
    """Thompson sampling: the arm with the highest Beta draw (see the module doc)."""
    draws = [rng.beta(1 + C * a["grade"]["accuracy"], 1 + C * (1 - a["grade"]["accuracy"]) + a["refined"])
             for a in pool]
    return int(np.argmax(draws))


def refine_parts(ev, group, arm, rng_examples, rng_cases):
    """What a refinement call shows: the group, the rule, and where it fails."""
    parts = R.group_prompt(ev, group, rng_examples)[:3 + 2 * N_SHOWN]      # header + N_SHOWN moves
    g = arm["grade"]
    parts.append(R.text_part(
        f"A rule written for this group, and how it did on ALL recorded moves of this level:\n\n"
        f"RULE: {arm['rule']}\nIt applies to {g['moves']} recorded moves ({g['in_group_moves']} of this "
        f"group's {g['group_moves']}), is exactly right on {g['accuracy']:.0%} of them, and predicts "
        f"{g['changes_right']} real changes exactly.\n```python\n{arm['code']}\n```"))
    wrong = arm["res"]["wrong"]
    shown = 0
    for j in rng_cases.permutation(len(wrong))[:N_FAILING]:
        k, pred = wrong[j]
        parts += [R.text_part(
            f"The rule was WRONG on this move: {R.move_text(ev, k)}\nWhere its prediction differs from "
            "the real board:\n  " + "\n  ".join(
                R._areas(pred, ev.after[k], ev.mask, lambda p, q, n: f"predicted {p}, really {q} x{n}", 12))
            + "\nPicture: BEFORE | THE RULE'S PREDICTION | REAL AFTER (cyan boxes: where the "
            "prediction is wrong)."),
            R.image_part(R.render_compare(ev.before[k], pred, ev.after[k], ev.mask))]
        shown += 1
    missed = g.get("missed") or []
    for j in rng_cases.permutation(len(missed))[:N_FAILING - shown]:
        k = int(missed[j])
        parts += [R.text_part("The rule did not apply to this move of the group, where the board DID "
                              f"change: {R.move_text(ev, k)}"),
                  R.image_part(R.render_move(ev.before[k], ev.after[k], ev.mask))]
        shown += 1
    parts.append(R.text_part(INSTRUCTION))
    return parts, shown


def ask(client, conv, step, ev, ev_path, group):
    """One refinement call: a thinking answer, a code-only retry if it ran out of
    room, one repair attempt if the code cannot run. Returns the child arm."""
    o = client.complete(conv, n=1, max_tokens=R.MAX_TOKENS[True], thinking=True,
                        seed=100 + step, **R.SAMPLING[True])[0]
    tokens, finished, retried = o["tokens"], o["finish"] == "stop", False
    code = R.extract_code(o["text"])
    child = (evaluate(code, ev, ev_path, group) if code else
             {"code": None, "stage": "no-code",
              "reason": "no ```python block in the answer" + ("" if finished else " (ran out of tokens)")})
    if (code is None and not finished) or child["stage"] in ("static", "runtime", "timeout"):
        msg = R.FINISH if code is None else R.REPAIR.format(reason=child["reason"])
        o2 = client.complete(conv + [{"role": "assistant", "content": R.tail(o["text"])},
                                     {"role": "user", "content": msg}],
                             n=1, max_tokens=R.MAX_TOKENS[False], thinking=False,
                             seed=150 + step, **R.SAMPLING[False])[0]
        tokens, retried = tokens + o2["tokens"], True
        code2 = R.extract_code(o2["text"])
        if code2:
            child = evaluate(code2, ev, ev_path, group)
    child.update(tokens=tokens, finished=finished, retried=retried)
    return child


def slim(arm):
    keep = ("code", "rule", "stage", "reason", "refined", "origin", "step", "parent", "tokens", "finished",
            "retried", "shown_failing", "seconds", "group_changing_covered", "v2", "transfer", "duplicate")
    out = {k: v for k, v in arm.items() if k in keep}
    if "grade" in arm:
        out["grade"] = dict(arm["grade"], missed=len(arm["grade"].get("missed") or []))
    return out


def run_group(game, group, client, out, calls):
    t0 = time.time()
    ev_path = os.path.join(V1, f"{game}_train.npz")
    ev = LevelEvidence.load(ev_path)
    keys = ev.keys_of(group)
    pool, n_v1 = starting_pool(game, group, ev, ev_path)
    best0 = max((a["grade"]["accuracy"] for a in pool), default=0.0)
    info = {"game": game, "group": group, "level": ev.level + 1,
            "moves": int(ev.count[keys].sum()), "cases": int(len(keys)),
            "changing_cases": int((ev.changed[keys] & ~ev.frozen[keys]).sum()),
            "conflict_cases": int((ev.n_variants[keys] > 1).sum()),
            "tier0a_candidates": n_v1, "starting_arms": len(pool), "starting_best_accuracy": best0}
    print(f"[{game} {group}] {info['moves']} moves, {info['changing_cases']} changing cases, "
          f"{info['conflict_cases']} with conflicting outcomes; {len(pool)} starting arms, best {best0:.1%}", flush=True)
    steps, exact_at, winner = [], None, None
    if client is not None and pool:
        rng = rng_for(game, group, "bandit")
        seen = {a["code"] for a in pool}
        for step in range(1, calls + 1):
            ts = time.time()
            i = pick(pool, rng)
            parent = pool[i]
            parts, shown = refine_parts(ev, group, parent, rng_for(game, group, "prompt"),
                                        rng_for(game, group, "cases", step))
            child = ask(client, R.conversation(parts, thinking=True), step, ev, ev_path, group)
            parent["refined"] += 1
            child.update(refined=0, origin="refined", step=step, parent=pool.index(parent),
                         shown_failing=shown, seconds=round(time.time() - ts))
            child["duplicate"] = bool(child["code"] and child["code"] in seen)
            steps.append(child)
            acc = child["grade"]["accuracy"] if "grade" in child else None
            print(f"[{game} {group}] step {step}: refined arm {child['parent']} "
                  f"({parent['grade']['accuracy']:.1%}) -> {child['stage']}"
                  + (f", {acc:.1%} on {child['grade']['moves']} moves" if acc is not None else f" ({child['reason'][:60]})")
                  + (" EXACT" if is_exact(child) else "") + f"; {child['tokens']} tokens, {child['seconds']}s", flush=True)
            if is_exact(child):
                exact_at, winner = step, child
                break
            if in_pool(child) and not child["duplicate"]:
                seen.add(child["code"])
                pool.append(child)
    everything = pool + [s for s in steps if s not in pool]
    checked = [a for a in everything if a["stage"] == "checked"]
    best = winner or max(checked, key=lambda a: (a["grade"]["plan_eligible"], a["grade"]["admitted"] and a["grade"]["exact_all"],
                                                 a["grade"]["accuracy"], a["grade"]["moves"]), default=None)
    t_path = os.path.join(V1, f"{game}_transfer.npz")
    if best is not None and os.path.exists(t_path):          # T0: the rule untouched on the next level
        r2 = check_rule(best["code"], t_path)
        best["transfer"] = ({"cases": int(r2["applies"].sum()), "right": int((r2["applies"] & r2["correct"]).sum())}
                            if r2["stage"] == "checked" else {"failed": r2.get("reason", r2["stage"])})
    res = dict(info, calls=len(steps), exact_at_step=exact_at, reached_exact=exact_at is not None,
               exact_up_to_conflicts=any(exact_up_to_conflicts(a) for a in checked),
               best=slim(best) if best else None, tokens=sum(s.get("tokens", 0) for s in steps),
               seconds=round(time.time() - t0), steps=[slim(s) for s in steps],
               pool=[slim(a) for a in pool])
    with open(os.path.join(out, f"{game}__{slug(group)}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def verdict(n_exact):
    if n_exact >= 3:
        return "3 or more exact: feedback does convert near-misses; CP1 keeps refinement central."
    if n_exact == 2:
        return "Exactly 2 exact: CP1 goes ahead as written."
    return ("1 or fewer exact: the misses are hidden state or edge cases; CP1 prioritises partial "
            "claims and api.t.")


def report(out, results, client):
    n_exact = sum(r["reached_exact"] for r in results)
    L = ["# Rulebook v2, CP0: near-miss refinement with a bandit", "",
         f"Pre-registered in docs/plans/rulebook-v2-prereg.md section 3. {len(results)} action groups, at most "
         f"{CALLS} refinement calls each, Tier 0a's frozen evidence and candidates. Tuned on dev games."]
    if client is not None:
        m = client.meter
        L += [f"Model `{client.served_model()}`, thinking on. LLM: {m['requests']} requests ({m['cached']} from "
              f"cache), {m['completions']} new answers, {m['tokens']:,} tokens, {m['seconds'] / 60:.0f} "
              f"request-minutes, {m['retries']} retried connections.", "",
              "## Result", "",
              f"**{n_exact} of {len(results)} groups reached an exact rule** (expected: 2-4).", "",
              verdict(n_exact)]
    L += ["", "| game | group | moves | changing cases | cases with conflicting outcomes | starting arms | best before | "
              "calls | best after | exact (at call) | exact up to conflicts | group's changing cases it covers | "
              "next level: applies / right |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|"]
    for r in results:
        b = r["best"] or {}
        g, t = b.get("grade") or {}, b.get("transfer") or {}
        L.append(f"| {r['game']} | {r['group']} | {r['moves']:,} | {r['changing_cases']} | {r['conflict_cases']} | "
                 f"{r['starting_arms']} | {r['starting_best_accuracy']:.1%} | {r['calls']} | "
                 f"{g.get('accuracy', 0):.1%} | {'**yes** (' + str(r['exact_at_step']) + ')' if r['reached_exact'] else 'no'} | "
                 f"{'yes' if r['exact_up_to_conflicts'] else 'no'} | {b.get('group_changing_covered', 0):.0%} | "
                 + (f"{t['cases']} / {t['right']}" if "cases" in t else "-") + " |")
    L += ["", "'Exact' is v1's plan-eligible: admitted, right on every case it applies to, none of them with "
              "conflicting outcomes. 'Exact up to conflicts': right on every case's usual outcome, but some of "
              "those cases have also been recorded ending differently, which no rule that reads only the "
              "screen and the action can get right."]
    for r in results:
        L += ["", f"## {r['game']} {r['group']}", ""]
        for s in r["steps"]:
            g = s.get("grade")
            L.append(f"- call {s['step']}: refined arm {s['parent']} -> {s['stage']}"
                     + (f", right on {g['accuracy']:.1%} of {g['moves']} moves, gain {g['gain']}"
                        f"{', EXACT' if g['plan_eligible'] else ''}" if g else f" ({s['reason'][:80]})")
                     + (", same code as an earlier rule" if s.get("duplicate") else "")
                     + f"; {s.get('tokens', 0):,} tokens" + (", retried" if s.get("retried") else ""))
        if r["best"]:
            L += ["", f"Best rule: **{r['best'].get('rule', '')}**"]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:24]))
    print(f"\nwrote {out}/report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rulebook", "v2", "nearmiss"))
    ap.add_argument("--url", default="http://127.0.0.1:8018/v1")
    ap.add_argument("--calls", type=int, default=CALLS)
    ap.add_argument("--dry-run", action="store_true", help="no LLM: print each group's starting pool")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    client = None if a.dry_run else ChatClient(a.url, cache_path=os.path.join(a.out, "llm_cache.jsonl"))
    if client is not None:
        print(f"model: {client.served_model()}", flush=True)
    results, failed = [], []

    def job(gg):
        try:
            return run_group(gg[0], gg[1], client, a.out, a.calls)
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(gg)
            return None
    with ThreadPoolExecutor(WORKERS) as ex:
        results = [r for r in ex.map(job, GROUPS) if r is not None]
    if failed:
        sys.exit(f"{len(failed)} group(s) failed: {failed}. Finished groups are saved in {a.out}; "
                 "rerun the same command to resume from the answer cache.")
    if not a.dry_run:
        report(a.out, results, client)


if __name__ == "__main__":
    main()
