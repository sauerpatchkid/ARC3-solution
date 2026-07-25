#!/usr/bin/env python3
"""analyze_curves.py — levels-vs-action-budget curves, AULC, and RHAE.

Post-processing only: reads the metrics.json that compute_metrics.py already
wrote for each run, so it never touches a corpus and can be re-run on every
historical run without re-playing anything.

What it produces, per (game, arm) group:
  * max_level(t) as a step function per seed, on a log action axis
  * the median curve across seeds with a min–max band       (levels_vs_budget.png)
  * AULC — see below                                        (scalar per group)
  * actions-to-level-k, median + how many seeds reached it   (censoring-aware)
  * RHAE per level, if --human-baselines is supplied

  python analyze_curves.py results/sweeps/sweep_<stamp>.manifest \\
      --out results/sweeps/curves_<stamp>

Input is either a sweep manifest (run_dir<TAB>game<TAB>seed<TAB>arm, what
sweep.sh writes) or one or more run directories via --runs.

------------------------------------------------------------------------------
AULC definition (state this verbatim in the write-up)
------------------------------------------------------------------------------
    AULC = (1 / ln(T_max/T_min)) * integral from T_min to T_max of L(t) d(ln t)

where L(t) is the max level completed by action t. It is the agent's average
completed level, weighted uniformly in LOG action budget, so early progress
counts as much as an equal-length later stretch. Units are levels; range is
[0, max_level]. It is NOT a standardized quantity, so the window matters:
T_min defaults to 100 and T_max to the SMALLEST n_actions in the group, so
every seed is integrated over an identical budget and no run is extrapolated
past where it actually stopped. Both bounds are printed with every number.

Rankings can flip with the budget — always report T_max next to the AULC.

------------------------------------------------------------------------------
RHAE
------------------------------------------------------------------------------
RHAE_level = (h_level / a_level)^2, where h is the human baseline action count
and a is the agent's; unsolved levels score 0. Pass --human-baselines with a
JSON file of {game: {level: human_actions}}.

Two honest caveats to carry into the report:
  1. The square makes this degenerate at our action counts — a level solved in
     23k actions against a ~50-action human baseline scores 5e-6. Everything
     rounds to 0.00 in a normal table, so the plain ratio h/a is printed next
     to it (same ranking, readable magnitude) and per-level ACTION COUNTS
     remain the primary figure.
  2. ARC states public-set scores are not a valid measure of progress. These
     are comparative agent diagnostics, not benchmark results.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

T_MIN_DEFAULT = 100.0


def load_manifest(path):
    """sweep.sh manifest: run_dir <TAB> game <TAB> seed <TAB> arm."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                rows.append({"run_dir": parts[0], "game": parts[1],
                             "seed": parts[2], "arm": parts[3]})
    return rows


def load_runs(run_dirs):
    """Infer (game, seed, arm) from each run's own metrics.json/run_config.json."""
    rows = []
    for d in run_dirs:
        mj = os.path.join(d, "metrics.json")
        if not os.path.exists(mj):
            continue
        with open(mj) as f:
            m = json.load(f)
        arm = "on"
        cfg = os.path.join(d, "run_config.json")
        if os.path.exists(cfg):
            with open(cfg) as f:
                arm = "on" if json.load(f).get("reset_on_level", True) else "off"
        rows.append({"run_dir": d, "game": m.get("game", "unknown"),
                     "seed": str(m.get("seed", "NA")), "arm": arm})
    return rows


def read_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        return json.load(f)


def level_at(levelups, t):
    """Max level completed by action t (levelups = [[action_num, level], ...])."""
    return sum(1 for a, _ in levelups if a <= t)


def aulc(levelups, t_min, t_max):
    """Exact integral of the L(t) step function over d(ln t), normalized.

    L is piecewise constant, so the integral is a closed-form sum over the
    segments between level-up actions — no grid, no interpolation error.
    """
    if t_max <= t_min:
        return 0.0
    pts = sorted(a for a, _ in levelups if t_min < a < t_max)
    edges = [t_min] + pts + [t_max]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            total += level_at(levelups, lo) * (math.log(hi) - math.log(lo))
    return total / math.log(t_max / t_min)


def actions_to_level(levelups, k):
    """Action index at which level k was completed, or None if never."""
    for a, _ in sorted(levelups):
        if level_at(levelups, a) >= k:
            return a
    return None


def summarize_group(runs, t_min):
    """runs = [(seed, metrics)]. Returns the per-group analysis dict."""
    budgets = [m["n_actions"] for _, m in runs]
    t_max = float(min(budgets))
    out = {"n_seeds": len(runs), "t_min": t_min, "t_max": t_max,
           "budgets": budgets, "aulc_per_seed": {}, "levels_per_seed": {},
           "actions_to_level": {}}
    for seed, m in runs:
        lus = m.get("levelup_events") or []
        out["aulc_per_seed"][seed] = round(aulc(lus, t_min, t_max), 4)
        out["levels_per_seed"][seed] = level_at(lus, t_max)
    vals = list(out["aulc_per_seed"].values())
    out["aulc_median"] = round(st.median(vals), 4)
    out["aulc_min"], out["aulc_max"] = min(vals), max(vals)
    max_k = max(out["levels_per_seed"].values(), default=0)
    for k in range(1, max_k + 1):
        hits = [(seed, actions_to_level(m.get("levelup_events") or [], k))
                for seed, m in runs]
        reached = [a for _, a in hits if a is not None]
        out["actions_to_level"][k] = {
            "reached": len(reached), "of": len(runs),
            "median": st.median(reached) if reached else None,
            "raw": sorted(reached),
        }
    return out


def rhae_table(group, baselines, game):
    """RHAE per level for one group. baselines = {game: {level: human_actions}}."""
    hb = (baselines or {}).get(game) or {}
    rows = []
    for k, info in sorted(group["actions_to_level"].items()):
        h = hb.get(str(k), hb.get(k))
        if h is None or info["median"] is None:
            rows.append({"level": k, "human": h, "agent": info["median"],
                         "ratio": None, "rhae": 0.0 if h is not None else None})
            continue
        ratio = h / info["median"]
        rows.append({"level": k, "human": h, "agent": info["median"],
                     "ratio": ratio, "rhae": ratio ** 2})
    return rows


def plot_curves(groups, t_min, path):
    """Median max-level curve per group with a min–max band, log action axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  (matplotlib unavailable — skipping plot)")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for (game, arm), runs in sorted(groups.items()):
        t_max = min(m["n_actions"] for _, m in runs)
        grid = np.logspace(math.log10(t_min), math.log10(t_max), 400)
        curves = np.array([[level_at(m.get("levelup_events") or [], t)
                            for t in grid] for _, m in runs], dtype=float)
        med = np.median(curves, axis=0)
        label = f"{game} (reset {arm}, n={len(runs)})"
        line, = ax.plot(grid, med, label=label, linewidth=2)
        if len(runs) > 1:
            ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0),
                            alpha=0.15, color=line.get_color())
    ax.set_xscale("log")
    ax.set_xlabel("cumulative environment actions (log)")
    ax.set_ylabel("max level completed")
    ax.set_title("Levels solved vs. action budget (median, min–max band)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="?", default=None,
                    help="sweep manifest (run_dir<TAB>game<TAB>seed<TAB>arm)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run dirs instead of a manifest (globs ok)")
    ap.add_argument("--out", default=None,
                    help="output prefix (writes <out>.md, <out>.csv, <out>.png)")
    ap.add_argument("--t-min", type=float, default=T_MIN_DEFAULT,
                    help=f"AULC lower bound (default {T_MIN_DEFAULT:g})")
    ap.add_argument("--human-baselines", default=None,
                    help="JSON {game: {level: human_actions}} to enable RHAE")
    a = ap.parse_args()

    if a.runs:
        dirs = [d for pat in a.runs for d in sorted(glob.glob(pat))]
        rows = load_runs(dirs)
    elif a.manifest:
        rows = load_manifest(a.manifest)
    else:
        raise SystemExit("give a manifest or --runs")
    if not rows:
        raise SystemExit("no runs found")

    baselines = None
    if a.human_baselines:
        with open(a.human_baselines) as f:
            baselines = json.load(f)

    groups = defaultdict(list)
    for r in rows:
        try:
            groups[(r["game"], r["arm"])].append((r["seed"], read_metrics(r["run_dir"])))
        except FileNotFoundError:
            print(f"  (no metrics.json in {r['run_dir']} — skipping)")

    out = a.out or os.path.join(os.getenv("EVAL_RESULTS_DIR", "results"),
                                "sweeps", "curves")
    outdir = os.path.dirname(out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    lines = ["# Levels vs. action budget", ""]
    lines.append("AULC = average completed level, weighted uniformly in log "
                 "action budget (see analyze_curves.py docstring). Units: "
                 "levels. Always read it next to T_max.")
    lines.append("")
    csv_rows = []

    for (game, arm), runs in sorted(groups.items()):
        g = summarize_group(runs, a.t_min)
        lines.append(f"## {game}  |  arm=reset_{arm}  |  {g['n_seeds']} seed(s)")
        lines.append(f"- budget window: T_min={g['t_min']:g}  T_max={g['t_max']:g} "
                     f"(smallest run in group; budgets={g['budgets']})")
        lines.append(f"- **AULC = {g['aulc_median']}** levels "
                     f"(min {g['aulc_min']}, max {g['aulc_max']}; "
                     f"per seed {g['aulc_per_seed']})")
        lines.append(f"- max level at T_max, per seed: {g['levels_per_seed']}")
        for k, info in sorted(g["actions_to_level"].items()):
            med = info["median"]
            lines.append(f"- actions to level {k}: "
                         f"{'median ' + format(med, 'g') if med is not None else 'never'}"
                         f"  ({info['reached']}/{info['of']} seeds reached)"
                         f"  raw={info['raw']}")
        row = {"game": game, "arm": arm, "n_seeds": g["n_seeds"],
               "t_min": g["t_min"], "t_max": g["t_max"],
               "aulc_median": g["aulc_median"], "aulc_min": g["aulc_min"],
               "aulc_max": g["aulc_max"]}
        for k, info in sorted(g["actions_to_level"].items()):
            row[f"actions_to_L{k}_median"] = info["median"]
            row[f"actions_to_L{k}_reached"] = f"{info['reached']}/{info['of']}"

        if baselines is not None:
            lines.append("")
            lines.append("| level | human h | agent a | ratio h/a | RHAE (h/a)² |")
            lines.append("|------:|--------:|--------:|----------:|------------:|")
            for r in rhae_table(g, baselines, game):
                hh = "—" if r["human"] is None else f"{r['human']:g}"
                aa = "never" if r["agent"] is None else f"{r['agent']:g}"
                rr = "—" if r["ratio"] is None else f"{r['ratio']:.4g}"
                vv = "—" if r["rhae"] is None else f"{r['rhae']:.3g}"
                lines.append(f"| {r['level']} | {hh} | {aa} | {rr} | {vv} |")
                row[f"rhae_L{r['level']}"] = r["rhae"]
            lines.append("")
            lines.append("_RHAE is reported for completeness; the square makes it "
                         "near-degenerate at these action counts, and ARC states "
                         "public-set scores are not a valid measure of progress. "
                         "Per-level action counts above are the primary figure._")
        lines.append("")
        csv_rows.append(row)

    png = plot_curves(groups, a.t_min, out + ".png")
    if png:
        lines.append(f"![levels vs budget]({os.path.basename(png)})")

    with open(out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    if csv_rows:
        cols = []
        for r in csv_rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)

    print("\n".join(lines))
    print(f"\nwrote {out}.md, {out}.csv" + (f", {png}" if png else ""))


if __name__ == "__main__":
    main()
