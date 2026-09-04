#!/usr/bin/env python3
"""compare.py — put two or more agents' benchmark runs side by side.

Takes the manifests written by `make bench` (one per agent) and produces a
per-game table of the metrics that matter, so "is mine better?" is a table
rather than an argument.

    uv run python compare.py results/sweeps/bench_standard_v1_goose_*.manifest \\
                             results/sweeps/bench_standard_v1_mine_*.manifest

    make compare M1=<manifest> M2=<manifest>

It refuses to compare manifests that did not run the same games and seeds,
because that is the mistake this whole benchmark setup exists to prevent.

HOW TO READ THE OUTPUT — the repo's standing rule is that no exploration
metric means anything on its own:

  levels        the headline. Reported as "median (k/n seeds reached >=1)".
                Most runs complete nothing, so the k/n is not optional.
  uniq/act      unique canonical states per action — the coverage number.
  novelty_late  new states per 1k actions over the final 10%. The stall
                detector: it flattens long before the level counter does.
  meaningful    change rate after decorative cells are masked. HIGH IS NOT
                GOOD ON ITS OWN — an agent jiggling an animation scores 1.00.
                Read it next to redundancy and coverage.
  redundancy    fraction of actions repeating a (state, action) already tried.
  act/s         throughput. A "better" agent that is 10x slower has not won.

Differences are shown against the FIRST manifest given, which is the baseline.
"""
import argparse
import os
import statistics as st
from collections import defaultdict

from summarize_overnight import load_manifest, load_metrics

# metric key -> (column header, higher_is_better or None, format)
METRICS = [
    ("levels_completed", "levels", True, "{:.1f}"),
    ("unique_states_per_action", "uniq/act", True, "{:.4f}"),
    ("novelty_late_per_1k", "novelty_late", True, "{:.1f}"),
    ("meaningful_change_rate", "meaningful", None, "{:.3f}"),
    ("redundancy", "redundancy", False, "{:.3f}"),
    ("actions_per_sec", "act/s", True, "{:.0f}"),
]


def load_arm(manifest_path):
    """Load one manifest -> {(game, seed): metrics dict} plus a label."""
    rows = load_manifest(manifest_path)
    out = {}
    missing = []
    for r in rows:
        try:
            out[(r["game"], r["seed"])] = load_metrics(r["run_dir"])
        except FileNotFoundError:
            missing.append(r["run_dir"])
    # bench manifests are named
    #   bench_<suite>_v<n>_<agent>_<YYYYmmdd>_<HHMMSS>.manifest
    # The stamp is TWO underscore-separated fields, and an agent name may
    # itself contain underscores, so take everything between them.
    base = os.path.basename(manifest_path).replace(".manifest", "")
    parts = base.split("_")
    label = base
    if base.startswith("bench_") and len(parts) >= 6:
        label = "_".join(parts[3:-2]) or base
    return label, out, missing


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+", help="one per agent; first is the baseline")
    ap.add_argument("--out", default=None, help="write the table to this .md file")
    ap.add_argument("--force", action="store_true",
                    help="compare even if the suites do not match (results will "
                         "not be comparable; you will be told what differs)")
    a = ap.parse_args()

    arms = []
    for m in a.manifests:
        label, data, missing = load_arm(m)
        if missing:
            print(f"!! {label}: {len(missing)} run(s) have no metrics.json "
                  f"(first: {missing[0]}) - they are excluded")
        if not data:
            raise SystemExit(f"{m}: no runs with metrics found")
        arms.append((label, data))

    # --- the guard: same games, same seeds, or the comparison is meaningless
    keysets = [set(d) for _, d in arms]
    if len(set(map(frozenset, keysets))) != 1:
        base = keysets[0]
        print("\n!! MANIFESTS DO NOT COVER THE SAME (game, seed) PAIRS.")
        for (label, _), ks in zip(arms, keysets):
            extra, short = ks - base, base - ks
            print(f"   {label}: {len(ks)} runs"
                  + (f", {len(extra)} not in baseline" if extra else "")
                  + (f", missing {len(short)} the baseline has" if short else ""))
        if not a.force:
            raise SystemExit(
                "   Refusing to compare. Run the same frozen suite on both "
                "(make bench SUITE=... AGENT=...), or pass --force to compare "
                "the intersection anyway.")
        common = set.intersection(*keysets)
        print(f"   --force: comparing the {len(common)} shared runs only.\n")
        arms = [(l, {k: v for k, v in d.items() if k in common}) for l, d in arms]

    games = sorted({g for _, d in arms for g, _ in d})
    n_seeds = len({s for _, d in arms for _, s in d})
    base_label = arms[0][0]

    lines = []
    lines.append(f"# Benchmark comparison ({n_seeds} seeds/game, "
                 f"baseline = **{base_label}**)\n")
    lines.append("Medians across seeds. `levels` shows k/n seeds that completed "
                 "at least one level — a median of 0.0 with 1/3 reached is a "
                 "very different result from 0/3.\n")

    for header_key, header, higher_better, fmt in METRICS:
        lines.append(f"\n## {header}"
                     + ("" if higher_better is None else
                        f"  ({'higher' if higher_better else 'lower'} is better)"))
        lines.append("")
        lines.append("| game | " + " | ".join(l for l, _ in arms) + " |")
        lines.append("|---|" + "---|" * len(arms))
        for g in games:
            cells = []
            base_val = None
            for i, (label, data) in enumerate(arms):
                vals = [m.get(header_key) for (gg, _), m in data.items() if gg == g]
                v = med(vals)
                if v is None:
                    cells.append("-")
                    continue
                cell = fmt.format(v)
                if header_key == "levels_completed":
                    reached = sum(1 for x in vals if x)
                    cell += f" ({reached}/{len(vals)})"
                if i == 0:
                    base_val = v
                elif base_val is not None and higher_better is not None:
                    d = v - base_val
                    if abs(d) > 1e-9:
                        good = (d > 0) == higher_better
                        cell += f" ({'+' if d > 0 else ''}{fmt.format(d)}"
                        cell += " ✓)" if good else " ✗)"
                cells.append(cell)
            lines.append(f"| {g} | " + " | ".join(cells) + " |")

    lines.append("\n---\n**Read these together, never alone.** A high "
                 "`meaningful` change rate only means the frame keeps changing; "
                 "an agent jiggling a decorative animation scores 1.00. Judge a "
                 "result by `levels` and `uniq/act`, with `redundancy` and "
                 "`act/s` as the cost side.\n")
    lines.append("Runs are not reproducible seed-for-seed (nondeterministic CUDA "
                 "kernels diverge the weights after the first training step), so "
                 "differences smaller than the seed-to-seed spread are noise.")

    text = "\n".join(lines)
    print("\n" + text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
