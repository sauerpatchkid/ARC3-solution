#!/usr/bin/env python3
"""rank.py — the upgrade screen's leaderboard.

    uv run python experiments/upgrade_screen/rank.py results/screen/<stamp>/manifest.tsv

Every arm is compared, paired by (game, seed), with the adopted agent
("novelty"); map arms are also compared with "map". Ranking, decided before
any screen ran:

  1. levels gained over "novelty", summed over all (game, seed) pairs;
  2. then paired wins minus losses;
  3. then the gain in AULC (levels weighted early on a log action axis, the
     same measure analyze_curves.py reports), which rewards getting there
     sooner - it matters most in a short screen.

An arm is PROMISING if it has more levels than "novelty", more paired wins
than losses, and no game where it is worse on every seed. Promising arms go
to a proper dev test (3 seeds x 100k actions, a rule written first). A
screen with 2 seeds and 50k actions can miss late gains and can be fooled by
luck on one seed; treat the order as a shortlist, not a verdict.
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_curves import aulc  # noqa: E402
from manifest import read_manifest  # noqa: E402


def load(manifest):
    runs = {}
    cap = 0
    for row in read_manifest(manifest, strict=True):
        rundir, game, seed, arm = row["run_dir"], row["game"], row["seed"], row["arm"]
        path = rundir if os.path.isabs(rundir) else os.path.join(ROOT, rundir)
        m = json.load(open(os.path.join(path, "metrics.json")))
        runs[(arm, game, seed)] = m
        cap = max(cap, m["n_actions"])
    return runs, cap


def compare(runs, arm, ref, t_max):
    pairs = sorted((g, s) for (a, g, s) in runs if a == arm and (ref, g, s) in runs)
    if not pairs:
        return None
    w = t = l = 0
    lv_arm = lv_ref = 0
    au = 0.0
    per_game = {}
    for g, s in pairs:
        x, y = runs[(ref, g, s)], runs[(arm, g, s)]
        lx, ly = x["levels_completed"], y["levels_completed"]
        lv_ref, lv_arm = lv_ref + lx, lv_arm + ly
        w, l = w + (ly > lx), l + (ly < lx)
        t += ly == lx
        au += aulc(y.get("levelup_events") or [], 100, t_max) - aulc(x.get("levelup_events") or [], 100, t_max)
        per_game.setdefault(g, []).append(ly - lx)
    worse = sorted(g for g, d in per_game.items() if all(v < 0 for v in d))
    return {"pairs": len(pairs), "levels": lv_arm, "ref_levels": lv_ref, "gain": lv_arm - lv_ref,
            "w": w, "t": t, "l": l, "aulc_gain": au, "worse_every_seed": worse}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit(__doc__)
    manifest = argv[0]
    from screen import ARMS
    runs, cap = load(manifest)
    arms = sorted({a for a, _, _ in runs}, key=lambda a: list(ARMS).index(a) if a in ARMS else 99)
    rows = []
    for arm in arms:
        c = compare(runs, arm, "novelty", cap) if arm != "novelty" else None
        cm = compare(runs, arm, "map", cap) if ARMS.get(arm, ({}, None))[1] == "map" else None
        lv = sum(m["levels_completed"] for (a, _, _), m in runs.items() if a == arm)
        n = sum(1 for (a, _, _) in runs if a == arm)
        promising = bool(c and c["gain"] > 0 and c["w"] > c["l"] and not c["worse_every_seed"])
        rows.append({"arm": arm, "what": ARMS.get(arm, ({}, None, ""))[2], "runs": n, "levels": lv,
                     "vs_novelty": c, "vs_map": cm, "promising": promising})
    ranked = sorted([r for r in rows if r["vs_novelty"]],
                    key=lambda r: (r["vs_novelty"]["gain"], r["vs_novelty"]["w"] - r["vs_novelty"]["l"],
                                   r["vs_novelty"]["aulc_gain"]), reverse=True)
    ref = next((r for r in rows if r["arm"] == "novelty"), None)

    out_dir = os.path.dirname(manifest)
    lines = [f"# Upgrade screen leaderboard", "",
             f"Manifest: `{manifest}`  ", f"Runs: {len(runs)}; action cap {cap}.  ",
             f"Reference 'novelty' (the adopted agent): {ref['levels'] if ref else '?'} levels "
             f"over {ref['runs'] if ref else '?'} runs.", "",
             "| rank | arm | what it changes | levels | gain vs novelty | better/same/worse | games worse on every seed | AULC gain | vs map (gain, W/T/L) | promising |",
             "|---:|---|---|---:|---:|---|---|---:|---|---|"]
    for i, r in enumerate(ranked, 1):
        c, cm = r["vs_novelty"], r["vs_map"]
        vm = f"{cm['gain']:+d}, {cm['w']}/{cm['t']}/{cm['l']}" if cm else "-"
        lines.append(f"| {i} | {r['arm']} | {r['what']} | {r['levels']} | {c['gain']:+d} | "
                     f"{c['w']}/{c['t']}/{c['l']} | {', '.join(c['worse_every_seed']) or '-'} | "
                     f"{c['aulc_gain']:+.2f} | {vm} | {'YES' if r['promising'] else ''} |")
    best = next((r for r in ranked if r["promising"]), None)
    lines += ["", ("**Best candidate:** " + best["arm"] + f" ({best['what']}). Next: a dev test at "
                   "3 seeds x 100k with a rule written first.") if best else
              "**No candidate beat the novelty agent on the screen's rule.**", ""]
    games = sorted({g for _, g, _ in runs})
    lines += ["## Levels per game (summed over seeds)", "",
              "| arm | " + " | ".join(games) + " |", "|---|" + "---:|" * len(games)]
    for arm in arms:
        cells = [str(sum(m["levels_completed"] for (a, g2, _), m in runs.items() if a == arm and g2 == g))
                 for g in games]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "leaderboard.md"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "leaderboard.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "arm", "levels", "gain_vs_novelty", "wins", "ties", "losses",
                     "worse_every_seed", "aulc_gain", "promising"])
        for i, r in enumerate(ranked, 1):
            c = r["vs_novelty"]
            wr.writerow([i, r["arm"], r["levels"], c["gain"], c["w"], c["t"], c["l"],
                         " ".join(c["worse_every_seed"]), round(c["aulc_gain"], 3), r["promising"]])
    print(text)


if __name__ == "__main__":
    main()
