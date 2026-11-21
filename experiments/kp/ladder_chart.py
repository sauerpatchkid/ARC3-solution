#!/usr/bin/env python3
"""ladder_chart.py - one picture of several agents at the same budget: levels by 5,000 actions, and when they came.

The problem it fixes: the project's agents were measured in different sweeps (the baseline and the adopted
Goose at 100,000 actions, SOT and the planner at 5,000), so no one table or figure shows the steps from the
first to the last side by side. This reads any recordings through their manifests, cuts each run at the
budget, and draws two panels: the levels each agent had completed by the budget (a column per agent, a dot
per seed), and the levels completed against actions taken (a line per agent, the mean over its seeds).

    uv run python experiments/kp/ladder_chart.py --out results/kp/ladder/chart1 \\
        --series "Baseline Goose=results/sweeps/sweep_20260917_224431.manifest:A0" \\
                 "Adopted Goose=results/confirm_upgrade/20260927_221915/manifest.tsv:mb_gated_att" \\
                 "SOT Goose=results/kp/test/20261119_181131/manifest.tsv:SOT" \\
                 "SOT Goose + E20 planner=results/kp/test/20261119_181131/manifest.tsv:SOT-KP" \\
        [--games a,b] [--budget 5000] [--title "..."] [--note "..."] [--colours 4,5,6]

A series is LABEL=MANIFEST:ARM; several MANIFEST:ARM joined by "+" pool their runs under one label (two runs
of the same agent). It writes <out>.png and <out>.csv (the same numbers as a table). It gives no verdict:
it only draws what the runs' metrics.json files say (levelup_events, on the actions-taken clock).

Colours and marks follow the chart rules the project's other figures use: a fixed colour order checked for
colour-blind separation, a 2-point line, a legend plus a label at each line's end, text in ink colours only.
An agent keeps its colour from one figure to the next: --colours gives each series its place in that order
(the baseline 1, the adopted Goose 2, SOT 3, SOT with the planner 4, with the explorer too 5) when a figure
leaves some of the agents out.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_curves import level_at  # noqa: E402
from levels_by_budget import load  # noqa: E402  (a manifest's runs: {(game, seed): (levelup_events, actions)})

# the reference palette's first six categorical slots, in their fixed order, on its light surface
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STEP = 25                   # the curve is drawn every this many actions


def read_series(text, games):
    """(label, {seed: [levelup actions of every game]}, runs per seed) for one LABEL=MANIFEST:ARM[+MANIFEST:ARM]."""
    label, specs = text.split("=", 1)
    by_seed, n_runs = {}, {}
    for i, spec in enumerate(specs.split("+")):
        for (game, seed), (ups, _) in load(spec, games).items():
            key = f"{seed}" if i == 0 else f"{seed} (run {i + 1})"      # a second run of a seed is its own dot
            by_seed.setdefault(key, []).extend(int(a) for a, _ in ups)
            n_runs[key] = n_runs.get(key, 0) + 1
    return label, dict(sorted(by_seed.items())), n_runs       # the seeds in one order in every column


def rounded_column(ax, x, height, width, colour, radius_pts=4):
    """A column with its top corners rounded and its foot square on the baseline. Call it once the
    axes have their final limits and place in the figure: the corner is sized in points."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    if height <= 0:
        return
    inv = ax.transData.inverted()
    px = radius_pts * ax.figure.dpi / 72
    (x0, y0), (x1, y1) = inv.transform((0, 0)), inv.transform((px, px))
    rx, ry = min(x1 - x0, width / 2), min(y1 - y0, height)
    lo, hi = x - width / 2, x + width / 2
    verts = [(lo, 0), (lo, height - ry), (lo, height), (lo + rx, height), (hi - rx, height), (hi, height),
             (hi, height - ry), (hi, 0), (lo, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), linewidth=0, facecolor=colour, zorder=3))


def two_lines(label):
    """An agent's name for the foot of its column: what follows " + " on a line of its own."""
    return "\n+ ".join(part.replace(" ", "\n", 1) if len(part) > 12 and "+" not in label else part
                       for part in label.split(" + "))


def draw(series, budget, out, title, note, n_games, colours):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.9), dpi=170, gridspec_kw={"width_ratios": [1, 1.45]})
    fig.patch.set_facecolor(SURFACE)
    longest = max(len(label) for label, _, _ in series) + 6         # a line's end label: its name and its value
    fig.subplots_adjust(left=0.07, right=min(0.84, 0.985 - 0.0066 * longest), top=0.78,
                        bottom=0.17 if note else 0.11, wspace=0.16)
    means = [sum(level_at([(a, 0) for a in ups], budget) for ups in by_seed.values()) / len(by_seed)
             for _, by_seed, _ in series]
    top = max(max(level_at([(a, 0) for a in ups], budget) for ups in by_seed.values()) for _, by_seed, _ in series)
    ymax = top * 1.16
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_ylim(0, ymax)

    # left: a column per agent (the mean over its seeds), a dot per seed, the mean on the cap
    left.set_xlim(-0.6, len(series) - 0.4)
    for i, (label, by_seed, _) in enumerate(series):
        rounded_column(left, i, means[i], 0.46, colours[i])
        totals = [level_at([(a, 0) for a in ups], budget) for ups in by_seed.values()]
        spread = [i + (k - (len(totals) - 1) / 2) * 0.11 for k in range(len(totals))]
        left.scatter(spread, totals, s=34, color=INK, edgecolor=SURFACE, linewidth=1.3, zorder=5)
        left.text(i, max(max(totals), means[i]) + ymax * 0.035, f"{means[i]:.1f}", ha="center", va="bottom",
                  fontsize=11, fontweight="bold", color=INK)
    left.set_xticks(range(len(series)))
    left.set_xticklabels([two_lines(label) for label, _, _ in series], color=INK2, fontsize=9)
    left.set_ylabel(f"Levels completed by {budget:,} actions\n(sum over {n_games} games, one seed)")
    left.set_title("By the end of the budget", loc="left", fontsize=11, color=INK, pad=10)

    # right: levels completed against actions taken, the mean over an agent's seeds
    ts = list(range(0, budget + 1, STEP))
    ends = []
    for i, (label, by_seed, _) in enumerate(series):
        ys = [sum(level_at([(a, 0) for a in ups], t) for ups in by_seed.values()) / len(by_seed) for t in ts]
        right.plot(ts, ys, color=colours[i], linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label,
                   zorder=4)
        right.scatter([ts[-1]], [ys[-1]], s=46, color=colours[i], edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ends.append((ys[-1], label))
    gap = ymax * 0.055                  # end labels: keep the order, push apart the ones that would touch
    ends.sort()
    placed = []
    for y, _ in ends:
        placed.append(y if not placed else max(y, placed[-1] + gap))
    first = 0                           # a group that was pushed apart is centred on its own points, so that
    while first < len(placed):          # no label's leader crosses the label under it
        last = first
        while last + 1 < len(placed) and placed[last + 1] > ends[last + 1][0] + 1e-9:
            last += 1
        shift = (placed[last] - ends[last][0]) / 2
        if first:
            shift = min(shift, placed[first] - placed[first - 1] - gap)
        for k in range(first, last + 1):
            placed[k] -= shift
        first = last + 1
    for (y, label), y_text in zip(ends, placed):
        right.annotate(f"{label}  {y:.1f}", xy=(budget, y), xytext=(budget * 1.025, y_text), va="center", fontsize=9,
                       color=INK2, arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8) if abs(y_text - y) > 1e-9
                       else None, annotation_clip=False)
    right.set_xlim(0, budget)
    right.set_xticks(range(0, budget + 1, 1000))
    right.set_xticklabels([f"{t:,}" for t in range(0, budget + 1, 1000)])
    right.set_xlabel("Actions taken")
    right.set_title("As the actions are spent (mean over seeds)", loc="left", fontsize=11, color=INK, pad=10)
    # one legend for both panels, in a row under the title: a column and a line of one colour are one agent
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.012, 0.935), ncol=len(series), frameon=False,
               fontsize=9, labelcolor=INK2, handlelength=1.6, columnspacing=1.8, borderaxespad=0)

    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=13, fontweight="bold", color=INK)
    if note:
        fig.text(0.012, 0.012, note, ha="left", va="bottom", fontsize=8, color=MUTED, wrap=True)
    fig.savefig(out + ".png", facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--series", nargs="+", required=True, help='LABEL=MANIFEST:ARM[+MANIFEST:ARM], in the order drawn')
    ap.add_argument("--out", required=True, help="path without an ending: <out>.png and <out>.csv are written")
    ap.add_argument("--games", default="", help="only these games (default: every game in the manifests)")
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--title", default="Levels completed by 5,000 actions")
    ap.add_argument("--note", default="")
    ap.add_argument("--colours", default="", help="each series' place in the colour order, 1 to 6 (default: 1,2,3,...)")
    a = ap.parse_args()
    if len(a.series) > len(SERIES):
        sys.exit(f"at most {len(SERIES)} series: more would need colours that cannot be told apart")
    places = [int(k) for k in a.colours.split(",") if k] or list(range(1, len(a.series) + 1))
    if len(places) != len(a.series) or len(set(places)) != len(places) or not all(1 <= k <= len(SERIES) for k in places):
        sys.exit(f"--colours takes one place for each series, each different, from 1 to {len(SERIES)}")
    os.chdir(ROOT)
    games = [g for g in a.games.split(",") if g] or None
    series = [read_series(s, games) for s in a.series]
    n_games = {n for _, _, runs in series for n in runs.values()}
    if len(n_games) != 1:
        sys.exit(f"the series do not cover the same number of games a seed: {sorted(n_games)}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    draw(series, a.budget, a.out, a.title, a.note, n_games.pop(), [SERIES[k - 1] for k in places])
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "seed", "levels_by_1000", "levels_by_2000", f"levels_by_{a.budget}"])
        for label, by_seed, _ in series:
            for seed, ups in by_seed.items():
                ev = [(x, 0) for x in ups]
                w.writerow([label, seed, level_at(ev, 1000), level_at(ev, 2000), level_at(ev, a.budget)])
    print(f"wrote {a.out}.png and {a.out}.csv")
    for label, by_seed, _ in series:
        totals = [level_at([(x, 0) for x in ups], a.budget) for ups in by_seed.values()]
        print(f"  {label:<28} seeds {', '.join(by_seed)}: {totals}  mean {sum(totals) / len(totals):.1f}")


if __name__ == "__main__":
    main()
