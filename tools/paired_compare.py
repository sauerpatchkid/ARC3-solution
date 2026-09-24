#!/usr/bin/env python3
"""paired_compare.py — paired (game, seed) comparison of two arms, possibly
from two different sweep manifests, with the pre-registered adoption rule.

    uv run python tools/paired_compare.py \\
        --base results/sweeps/sweep_20260923_214629.manifest:A1 \\
        --new  results/sweeps/sweep_20260923_214629.manifest:A4

Each side is MANIFEST:ARM (the manifest's 4th column). Only (game, seed)
pairs present on both sides are compared. Reports per game: levels per seed,
paired wins/ties/losses, mean unique states, median late novelty and, when
the new arm used the return map, its share of actions and route outcomes.

Adoption rule (docs/plans/option-1-return-map.md; the same shape Plan B used):
  1. levels summed over the pairs: new >= base, and paired wins > losses;
  2. no game where new completes fewer levels than base on every seed.

Comparing across manifests is valid only when the base arm's code path is
unchanged between the two sweeps (for A1 it is: with EVAL_RETURN_MAP unset the
agent takes the same actions as before the map existed). Runs are not
byte-reproducible past ~300 actions (CUDA), so this is a distribution
comparison over seeds, never a per-trajectory one.
"""
import argparse
import json
import os
import statistics as st
import sys


def load(spec):
    path, arm = spec.rsplit(":", 1)
    out = {}
    with open(path) as f:
        for line in f:
            rundir, game, seed, a = line.rstrip("\n").split("\t")
            if a != arm:
                continue
            m = json.load(open(os.path.join(rundir, "metrics.json")))
            ms_path = os.path.join(rundir, "return_map_stats.json")
            out[(game, seed)] = dict(
                lv=m["levels_completed"], u=m["unique_states"], n=m["n_actions"],
                late=m.get("novelty_late_per_1k") or 0.0,
                aps=m.get("actions_per_sec") or 0.0,
                ms=json.load(open(ms_path)) if os.path.exists(ms_path) else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="MANIFEST:ARM for the reference arm")
    ap.add_argument("--new", required=True, help="MANIFEST:ARM for the candidate arm")
    ap.add_argument("--out", default="", help="optional .md path for the table")
    a = ap.parse_args()
    base, new = load(a.base), load(a.new)
    pairs = sorted(set(base) & set(new))
    if not pairs:
        sys.exit("no (game, seed) pairs in common")
    games = sorted({g for g, _ in pairs})

    lines = ["| game | seeds | base levels | new levels | W/T/L | uniq base | uniq new | late base | late new | map share | routes done/abandoned |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    W = T = L = 0
    tot_b = tot_n = 0
    worse_everywhere = []
    for g in games:
        seeds = sorted(s for gg, s in pairs if gg == g)
        lb = [base[(g, s)]["lv"] for s in seeds]
        ln = [new[(g, s)]["lv"] for s in seeds]
        w = sum(y > x for x, y in zip(lb, ln))
        l = sum(y < x for x, y in zip(lb, ln))
        t = len(seeds) - w - l
        W, T, L = W + w, T + t, L + l
        tot_b, tot_n = tot_b + sum(lb), tot_n + sum(ln)
        if l == len(seeds):
            worse_everywhere.append(g)
        ub = st.mean(base[(g, s)]["u"] for s in seeds)
        un = st.mean(new[(g, s)]["u"] for s in seeds)
        lab = st.median(base[(g, s)]["late"] for s in seeds)
        lan = st.median(new[(g, s)]["late"] for s in seeds)
        ms = [new[(g, s)]["ms"] for s in seeds]
        if all(m is not None for m in ms):
            share = f"{st.mean(m['map_actions'] / new[(g, s)]['n'] for m, s in zip(ms, seeds)):.0%}"
            routes = f"{sum(m['routes_completed'] for m in ms)}/{sum(m['routes_aborted'] for m in ms)}"
        else:
            share, routes = "-", "-"
        lines.append(f"| {g} | {len(seeds)} | {lb} | {ln} | {w}/{t}/{l} | {ub:.0f} | {un:.0f} | "
                     f"{lab:.1f} | {lan:.1f} | {share} | {routes} |")

    rule1 = tot_n >= tot_b and W > L
    rule2 = not worse_everywhere
    aps_b = st.mean(v["aps"] for k, v in base.items() if k in pairs)
    aps_n = st.mean(v["aps"] for k, v in new.items() if k in pairs)
    lines += ["",
              f"Pairs compared: {len(pairs)} over {len(games)} games.",
              f"Levels summed: base {tot_b}, new {tot_n}.",
              f"Paired (game, seed): new better {W}, same {T}, worse {L}.",
              f"Games where new is worse on every seed: {', '.join(worse_everywhere) or 'none'}.",
              f"Throughput: base {aps_b:.1f} act/s, new {aps_n:.1f} act/s.",
              "",
              f"Rule 1 (new >= base in total, wins > losses): {'PASS' if rule1 else 'FAIL'}",
              f"Rule 2 (no game worse on every seed): {'PASS' if rule2 else 'FAIL'}",
              f"Verdict: {'ADOPT' if rule1 and rule2 else 'DO NOT ADOPT'}"]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"# Paired comparison\n\nbase: `{a.base}`  \nnew: `{a.new}`\n\n{text}\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
