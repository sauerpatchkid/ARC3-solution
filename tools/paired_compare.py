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

--rule picks which pre-registered rule the verdict uses. Part 2 is the same in
all three; part 1 is:
  adopt    (default) levels new >= base, wins >  losses   Plan B, Option 1, upgrades
  dev      levels new >= base, wins >= losses             Rulebook dev gates (G3)
  confirm  levels new >  base, wins >  losses             Rulebook 25-game confirms
With --rule left out, verdicts are computed and printed exactly as before.

Added for the Coach plan (docs/plans/llm-coach.md §6), both off the rule's path:
  --expect N  blocks the verdict (BLOCKED, exit code 2) unless both arms have
              exactly the N declared (game, seed) runs, one each, the same
              action cap, and no run that ended on an error (run_end.json).
  a 95% bootstrap interval for the level delta, resampling games, printed as
  information only. With neither used, verdicts are computed exactly as before.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from manifest import read_manifest  # noqa: E402


def load(spec, problems=None):
    """{(game, seed): run summary} for one MANIFEST:ARM. Duplicate (game, seed)
    rows and runs without a metrics.json are reported into `problems` (used by
    --expect) instead of being silently overwritten or crashing."""
    path, arm = spec.rsplit(":", 1)
    problems = [] if problems is None else problems
    out = {}
    for row in read_manifest(path, strict=True):
        rundir, game, seed = row["run_dir"], row["game"], row["seed"]
        if row["arm"] != arm:
            continue
        if (game, seed) in out:
            problems.append(f"{spec}: duplicate run for {game} seed {seed}")
        mpath = os.path.join(rundir, "metrics.json")
        if not os.path.exists(mpath):
            problems.append(f"{spec}: no metrics.json for {game} seed {seed}")
            continue
        m = json.load(open(mpath))
        ms_path = os.path.join(rundir, "return_map_stats.json")
        cfg_path = os.path.join(rundir, "run_config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        out[(game, seed)] = dict(
            lv=m["levels_completed"], u=m["unique_states"], n=m["n_actions"],
            late=m.get("novelty_late_per_1k") or 0.0,
            aps=m.get("actions_per_sec") or 0.0,
            cap=cfg.get("max_actions"),
            end=m.get("termination"),      # None for runs before run_end.json
            ms=json.load(open(ms_path)) if os.path.exists(ms_path) else None)
    return out


def expect_check(n, base, new, pairs, problems):
    """--expect N: the verdict counts only if both arms have exactly the N
    declared (game, seed) runs, paired one to one, with the same action cap,
    and no run ended on an error. Returns the list of reasons it fails."""
    bad = list(problems)
    for name, side in (("base", base), ("new", new)):
        if len(side) != n:
            bad.append(f"{name} has {len(side)} runs, expected {n}")
    if len(pairs) != n:
        bad.append(f"{len(pairs)} (game, seed) pairs in common, expected {n}")
    caps = {v["cap"] for side in (base, new) for v in side.values()}
    if len(caps) > 1:
        bad.append(f"action caps differ across runs: {sorted(caps, key=str)}")
    for name, side in (("base", base), ("new", new)):
        for (g, s), v in sorted(side.items()):
            if v["end"] not in (None, "cap", "win"):
                bad.append(f"{name} {g} seed {s} ended with '{v['end']}'")
    return bad


# part 1 of each rule: (levels test, wins test, how it is printed)
RULES = {
    "adopt":   (lambda n, b: n >= b, lambda w, l: w > l,  "new >= base in total, wins > losses"),
    "dev":     (lambda n, b: n >= b, lambda w, l: w >= l, "new >= base in total, wins >= losses"),
    "confirm": (lambda n, b: n > b,  lambda w, l: w > l,  "new > base in total, wins > losses"),
}


def rule_one(rule, tot_n, tot_b, wins, losses):
    levels_ok, wins_ok, _ = RULES[rule]
    return levels_ok(tot_n, tot_b) and wins_ok(wins, losses)


def bootstrap_delta(per_game_delta, reps=10000, seed=0):
    """95% interval for the summed level delta, resampling GAMES with
    replacement (each game keeps all its seeds). Fixed seed: reruns agree."""
    import random
    rng = random.Random(seed)
    d = list(per_game_delta)
    sums = sorted(sum(rng.choice(d) for _ in d) for _ in range(reps))
    return sums[int(0.025 * reps)], sums[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="MANIFEST:ARM for the reference arm")
    ap.add_argument("--new", required=True, help="MANIFEST:ARM for the candidate arm")
    ap.add_argument("--out", default="", help="optional .md path for the table")
    ap.add_argument("--expect", type=int, default=0, metavar="N",
                    help="require exactly N valid (game, seed) pairs, one run each, same "
                         "action cap, no run ended on an error; otherwise the verdict is "
                         "BLOCKED (exit code 2). Off by default, so earlier verdicts stand.")
    ap.add_argument("--rule", choices=sorted(RULES), default="adopt",
                    help="which pre-registered rule the verdict uses (see the top of this file)")
    a = ap.parse_args()
    problems = []
    base, new = load(a.base, problems), load(a.new, problems)
    pairs = sorted(set(base) & set(new))
    if not pairs:
        sys.exit("no (game, seed) pairs in common")
    games = sorted({g for g, _ in pairs})

    lines = ["| game | seeds | base levels | new levels | W/T/L | uniq base | uniq new | late base | late new | map share | routes done/abandoned |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    W = T = L = 0
    tot_b = tot_n = 0
    worse_everywhere = []
    game_delta = []
    for g in games:
        seeds = sorted(s for gg, s in pairs if gg == g)
        lb = [base[(g, s)]["lv"] for s in seeds]
        ln = [new[(g, s)]["lv"] for s in seeds]
        w = sum(y > x for x, y in zip(lb, ln))
        l = sum(y < x for x, y in zip(lb, ln))
        t = len(seeds) - w - l
        W, T, L = W + w, T + t, L + l
        tot_b, tot_n = tot_b + sum(lb), tot_n + sum(ln)
        game_delta.append(sum(ln) - sum(lb))
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

    rule1 = rule_one(a.rule, tot_n, tot_b, W, L)
    rule2 = not worse_everywhere
    aps_b = st.mean(v["aps"] for k, v in base.items() if k in pairs)
    aps_n = st.mean(v["aps"] for k, v in new.items() if k in pairs)
    lo, hi = bootstrap_delta(game_delta)
    blocked = expect_check(a.expect, base, new, pairs, problems) if a.expect else []
    verdict = "ADOPT" if rule1 and rule2 else "DO NOT ADOPT"
    if blocked:
        verdict = "BLOCKED (run set incomplete or invalid; see above)"
    lines += ["",
              f"Pairs compared: {len(pairs)} over {len(games)} games.",
              f"Levels summed: base {tot_b}, new {tot_n}.",
              f"Level delta (new - base): {tot_n - tot_b:+d}; 95% bootstrap interval over "
              f"games: [{lo:+d}, {hi:+d}] (descriptive; not part of the rule).",
              f"Paired (game, seed): new better {W}, same {T}, worse {L}.",
              f"Games where new is worse on every seed: {', '.join(worse_everywhere) or 'none'}.",
              f"Throughput: base {aps_b:.1f} act/s, new {aps_n:.1f} act/s.",
              ""]
    if problems and not a.expect:
        lines += ["Warnings (use --expect N to block the verdict on these):"]
        lines += [f"  - {p}" for p in problems[:20]]
    if a.expect:
        lines += [f"Run-set check (--expect {a.expect}): "
                  f"{'PASS' if not blocked else 'FAIL'}"]
        lines += [f"  - {r}" for r in blocked[:20]]
        if len(blocked) > 20:
            lines.append(f"  - ... and {len(blocked) - 20} more")
    lines += [f"Rule 1 ({RULES[a.rule][2]}): {'PASS' if rule1 else 'FAIL'}",
              f"Rule 2 (no game worse on every seed): {'PASS' if rule2 else 'FAIL'}",
              f"Verdict: {verdict}"]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"# Paired comparison\n\nbase: `{a.base}`  \nnew: `{a.new}`\n\n{text}\n")
        print(f"\nwrote {a.out}")
    if blocked:
        sys.exit(2)


if __name__ == "__main__":
    main()
