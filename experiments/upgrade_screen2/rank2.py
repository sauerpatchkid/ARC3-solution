#!/usr/bin/env python3
"""rank2.py — round 2's leaderboard.

    uv run python experiments/upgrade_screen2/rank2.py results/screen2/<stamp>/manifest.tsv

Reuses round 1's scoring (experiments/upgrade_screen/rank.py: load, compare,
AULC) without editing it. Every arm is paired by (game, seed) with round 1's
best ("map_bars", run fresh here) and with the adopted agent ("novelty").

Pre-registered (docs/plans/upgrade-screen2.md, written before round 2 ran):
  rank by levels gained over map_bars, then paired wins minus losses, then AULC
  gain. An arm IMPROVES ON ROUND 1 if, against map_bars, it has more levels,
  more paired wins than losses, and no game worse on both seeds. Results on
  the 3 games round 1 never used (sp80, cn04, bp35) are reported separately as
  the check that a winner is not tuned to round 1's games.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "upgrade_screen"))

import rank  # noqa: E402  round 1's scoring, used as a library

BEST1 = "map_bars"
HELD_OUT = {"sp80", "cn04", "bp35"}


def _subset(runs, games):
    return {k: v for k, v in runs.items() if k[1] in games}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(__doc__)
    manifest = argv[0]
    import screen2  # noqa: F401  (sets round 2's arms on the shared runner module)
    from screen import ARMS
    runs, cap = rank.load(manifest)
    arms = [a for a in ARMS if any(k[0] == a for k in runs)]
    games = sorted({g for _, g, _ in runs})
    rows = []
    for arm in arms:
        vb = rank.compare(runs, arm, BEST1, cap) if arm != BEST1 else None
        vn = rank.compare(runs, arm, "novelty", cap) if arm != "novelty" else None
        held = rank.compare(_subset(runs, HELD_OUT), arm, BEST1, cap) if arm != BEST1 else None
        lv = sum(m["levels_completed"] for (a, _, _), m in runs.items() if a == arm)
        improves = bool(vb and vb["gain"] > 0 and vb["w"] > vb["l"] and not vb["worse_every_seed"])
        rows.append(dict(arm=arm, what=ARMS[arm][2], levels=lv, vb=vb, vn=vn, held=held, improves=improves))
    ranked = sorted([r for r in rows if r["vb"]],
                    key=lambda r: (r["vb"]["gain"], r["vb"]["w"] - r["vb"]["l"], r["vb"]["aulc_gain"]),
                    reverse=True)
    best = next((r for r in rows if r["arm"] == BEST1), None)
    nov = next((r for r in rows if r["arm"] == "novelty"), None)

    def f(c):
        return f"{c['gain']:+d}, {c['w']}/{c['t']}/{c['l']}" if c else "-"

    L = ["# Upgrade screen, round 2 - leaderboard", "",
         f"Manifest: `{manifest}`  ", f"Runs: {len(runs)}; action cap {cap}.  ",
         f"Round 1's best (map_bars): {best['levels'] if best else '?'} levels; "
         f"the adopted agent (novelty): {nov['levels'] if nov else '?'} levels.", "",
         "| rank | arm | what | levels | vs map_bars: gain, W/T/L | worse on every seed vs map_bars | AULC gain vs map_bars | vs novelty: gain, W/T/L | held-out games vs map_bars | improves on round 1 |",
         "|---:|---|---|---:|---|---|---:|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        vb = r["vb"]
        L.append(f"| {i} | {r['arm']} | {r['what']} | {r['levels']} | {f(vb)} | "
                 f"{', '.join(vb['worse_every_seed']) or '-'} | {vb['aulc_gain']:+.2f} | {f(r['vn'])} | "
                 f"{f(r['held'])} | {'YES' if r['improves'] else ''} |")
    win = next((r for r in ranked if r["improves"]), None)
    L += ["", ("**Improves on round 1:** " + win["arm"] + f" ({win['what']}).") if win else
          "**Nothing in round 2 beat round 1's best (map_bars) on the rule.**", "",
          "## Levels per game (summed over seeds)", "",
          "| arm | " + " | ".join(games) + " |", "|---|" + "---:|" * len(games)]
    for arm in arms:
        L.append(f"| {arm} | " + " | ".join(
            str(sum(m["levels_completed"] for (a, g2, _), m in runs.items() if a == arm and g2 == g))
            for g in games) + " |")
    text = "\n".join(L) + "\n"
    out = os.path.dirname(manifest)
    with open(os.path.join(out, "leaderboard.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(out, "leaderboard.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "arm", "levels", "gain_vs_map_bars", "wins", "ties", "losses",
                    "worse_every_seed", "aulc_gain", "gain_vs_novelty", "improves"])
        for i, r in enumerate(ranked, 1):
            vb, vn = r["vb"], r["vn"]
            w.writerow([i, r["arm"], r["levels"], vb["gain"], vb["w"], vb["t"], vb["l"],
                        " ".join(vb["worse_every_seed"]), round(vb["aulc_gain"], 3),
                        vn["gain"] if vn else "", r["improves"]])
    print(text)


if __name__ == "__main__":
    main()
