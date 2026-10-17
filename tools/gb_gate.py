#!/usr/bin/env python3
"""gb_gate.py - a gate's verdict from a finished run, as its pre-registration section defines it.

    uv run python tools/gb_gate.py g1 --run g1 --base g0b_bars

G1 (docs/3-glassbox/glassbox-prereg.md section 4). For each of the 8 dev games,
from the run's guard-fitter summary (s3_check/<game>_L<k>.guards.json, "after":
S3's book from S2's candidates plus every narrowed rule the fitter rescued):
  primary  the book's cell coverage on the training level is HIGHER than the
           base run's (G0b: CP1's rules on the same evidence), on >= 5 of 8 games
  safety   T1 is wrong on <= 5% of the transfer level's changing cases, on
           every game
  PASS     primary and safety.
Reported beside it, not gating: CP1's own criterion (T1 beats "nothing" and
"memory" with both intervals above zero), case coverage, trusted rules, the
LLM's calls, tokens and answers that ran out of room, H1 use, guard rescues.
Writes <run>/g1_verdict.json and <run>/g1_verdict.md; exit status 0 either way.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

import tiers  # noqa: E402
from gb import artifacts as A  # noqa: E402

PRIMARY_GAMES, SAFETY_WRONG = 5, 0.05


def summary(run_dir, game):
    p = glob.glob(os.path.join(run_dir, "s3_check", f"{game}_L*.guards.json"))
    if not p:
        raise FileNotFoundError(f"{run_dir}: no guard-fitter summary for {game}")
    with open(p[0]) as f:
        return json.load(f)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def game_numbers(run_dir, game):
    s = summary(run_dir, game)
    book, t1 = s["book"]["after"], (s.get("t1") or {}).get("after")
    rules = jsonl(os.path.join(run_dir, "s3_check", game, "rules.jsonl"))
    out = {"cell_coverage": book["cell_coverage"], "case_coverage": book["case_coverage"],
           "book_rules": len(book["members"]), "training_wrong": book["wrong"],
           "trusted": sum(r["v2_trusted"] for r in rules) + len(s["rescued"]),
           "admitted": sum(r["v2_admitted"] for r in rules), "guard_rescues_after": len(s["rescued"])}
    if t1:
        pm = t1["per_move"]
        out.update(t1_rate=pm["rate"]["book"], t1_nothing=pm["rate"]["nothing"], t1_memory=pm["rate"]["memory"],
                   t1_wrong=t1["wrong"], t1_vs_nothing=pm["diff"]["nothing"], t1_vs_memory=pm["diff"]["memory"],
                   beats_both=pm["beats_both"], t1_rules=len(t1["members"]))
    llm = os.path.join(run_dir, "s2_write", game, "llm.json")
    m = json.load(open(llm)) if os.path.exists(llm) else {}
    if "calls" in m:                                            # written by S2 (an imported run's is CP1's)
        cands = jsonl(os.path.join(run_dir, "s2_write", game, "candidates.jsonl"))
        checked = [c for c in cands if c.get("stage") == "checked"]
        trusted = [c for c in checked if c["trust"]["trusted"]]
        out.update(llm_calls=m["calls"], candidates=m["candidates"], tokens=m["tokens"], ran_out=m["ran_out"],
                   guard_rescues_in_loop=sum(c.get("origin") == "guard" and c["trust"]["trusted"] for c in checked),
                   with_primitives=round(sum(bool(c.get("primitives")) for c in checked) / len(checked), 3)
                   if checked else 0.0,
                   trusted_with_primitives=sum(bool(c.get("primitives")) for c in trusted),
                   failed_groups=[g["group"] for g in m["groups"] if g.get("error")])
    return out


def g1(run, base):
    games = list(tiers.TIERS["dev"])
    rows = {g: {"g1": game_numbers(run.dir, g), "base": game_numbers(base.dir, g)} for g in games}
    higher = [g for g in games if rows[g]["g1"]["cell_coverage"] > rows[g]["base"]["cell_coverage"]]
    unsafe = [g for g in games if rows[g]["g1"].get("t1_wrong", 0.0) > SAFETY_WRONG]
    primary, safety = len(higher) >= PRIMARY_GAMES, not unsafe
    return {"gate": "G1", "run": run.name, "base": base.name, "games": rows, "higher": higher,
            "primary": primary, "unsafe": unsafe, "safety": safety, "pass": primary and safety,
            "beats_both": [g for g in games if rows[g]["g1"].get("beats_both")],
            "failed_groups": {g: rows[g]["g1"].get("failed_groups") for g in games if rows[g]["g1"].get("failed_groups")}}


def table(v):
    lines = [f"# G1 verdict: {'PASS' if v['pass'] else 'FAIL'}",
             "",
             f"Primary (cell coverage above {v['base']}'s on >= {PRIMARY_GAMES} of 8): {len(v['higher'])} of 8 "
             f"({', '.join(v['higher']) or 'none'}): {'met' if v['primary'] else 'not met'}.  "
             f"Safety (T1 wrong <= {SAFETY_WRONG:.0%} on every game): "
             + ("met" if v["safety"] else f"NOT met on {', '.join(v['unsafe'])}") + ".",
             "",
             "| Game | Cell coverage (base -> G1) | Case coverage | Trusted | T1 per move (nothing, memory) | T1 wrong | "
             "Beats both | Calls | Tokens |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g, r in v["games"].items():
        a, b = r["g1"], r["base"]
        t1 = (f"{a['t1_rate']:.1%} ({a['t1_nothing']:.1%}, {a['t1_memory']:.1%})" if "t1_rate" in a else "-")
        lines.append(f"| {g} | {b['cell_coverage']:.1%} -> {a['cell_coverage']:.1%} | {b['case_coverage']:.1%} -> "
                     f"{a['case_coverage']:.1%} | {b['trusted']} -> {a['trusted']} | {t1} | "
                     f"{a.get('t1_wrong', 0):.1%} | {'yes' if a.get('beats_both') else 'no'} | "
                     f"{a.get('llm_calls', '-')} | {a.get('tokens', 0):,} |")
    if v["failed_groups"]:
        lines += ["", f"Groups that failed: {v['failed_groups']}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gate", choices=["g1"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    root = A.results_dir()
    run, base = (type("R", (), {"name": n, "dir": os.path.join(root, n)}) for n in (a.run, a.base))
    v = g1(run, base)
    with open(os.path.join(run.dir, "g1_verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    text = table(v)
    with open(os.path.join(run.dir, "g1_verdict.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
