#!/usr/bin/env python3
"""sweep_table.py - one table of a sweep's runs: levels per game, seed and arm, when they came, and who cleared them.

The problem it fixes: kp_verdict.py's tables are the registered test's and know its three arms only. The
sweeps made after it (the ladder, the explorer's dev sweep) have other arms. This prints the same kind of
table for any arms of any manifests, and gives no verdict.

    uv run python experiments/kp/sweep_table.py --arms \\
        "SOT-KP=results/kp/dev_e21/<sweep>/manifest.tsv:SOT-KP" \\
        "SOT-KP21=results/kp/dev_e21/<sweep>/manifest.tsv:SOT-KP21" [--games a,b] [--out table.md] [--by-game]

An arm is NAME=MANIFEST:ARM. Levels and level-up actions come from each run's metrics.json; for a run
made with EVAL_KP=1, who cleared each level comes from its kp/kp_stats.json (the planner E20, the
explorer E21, or Goose). Paired counts are against the first arm. For the levels-by-budget interval and
the level races, use tools/levels_by_budget.py on the same manifests.

--by-game prints the short form instead: one row per game, and for each arm the levels it completed and
the action at which the last of them was completed, written seed by seed ("6 / 6" and "793 / 763").
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from manifest import read_manifest  # noqa: E402  (the repo's one manifest reader)

BUDGETS = (1000, 2000, 5000)


def load(spec, games):
    """{(game, seed): dict(ups, by planning, by explorer, errors, explorer moves)} for one MANIFEST:ARM."""
    path, arm = spec.rsplit(":", 1)
    runs = {}
    for row in read_manifest(path, strict=True):
        if row["arm"] != arm or (games and row["game"] not in games):
            continue
        d = row["run_dir"] if os.path.isabs(row["run_dir"]) else os.path.join(ROOT, row["run_dir"])
        with open(os.path.join(d, "metrics.json")) as f:
            ups = [int(a) for a, _ in json.load(f).get("levelup_events") or []]
        stats = {}
        if os.path.exists(os.path.join(d, "kp", "kp_stats.json")):
            with open(os.path.join(d, "kp", "kp_stats.json")) as f:
                stats = json.load(f)
        runs[row["game"], int(row["seed"])] = dict(
            ups=ups, planning=stats.get("levels_by_planning", 0), explorer=stats.get("levels_by_explorer", 0),
            errors=stats.get("kp_errors", 0), explorer_moves=(stats.get("moves") or {}).get("explorer", 0),
            seconds=stats.get("reference_seconds", 0.0))
    return runs


def by_game(arms, pairs):
    """One row per game: for each arm, levels completed and the action of the last level-up, seed by seed."""
    md = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    seeds = sorted({s for _, s in pairs})
    lines = ["Each cell is written seed by seed (" + " / ".join(f"seed {s}" for s in seeds) + "). Moves: the action "
             "at which the run's last level was completed; `-` where no level was.", "",
             md(["Game"] + [f"{name}: {what}" for name, _ in arms for what in ("levels", "moves")]),
             md(["---"] * (1 + 2 * len(arms)))]
    for g in sorted({g for g, _ in pairs}):
        cells = []
        for _, r in arms:
            runs = [r[g, s] for s in seeds if (g, s) in pairs]
            cells += [" / ".join(str(len(x["ups"])) for x in runs),
                      " / ".join(f"{x['ups'][-1]:,}" if x["ups"] else "-" for x in runs)]
        lines.append(md([g] + cells))
    totals = []
    for _, r in arms:
        totals += [" / ".join(str(sum(len(r[g, s]["ups"]) for g, s in pairs if s == seed)) for seed in seeds), ""]
    lines.append(md([f"**All {len({g for g, _ in pairs})} games**"] + [f"**{t}**" if t else "" for t in totals]))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+", required=True, help="NAME=MANIFEST:ARM, the first is the one the others are paired with")
    ap.add_argument("--games", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--by-game", action="store_true", help="the short form: one row per game")
    a = ap.parse_args()
    os.chdir(ROOT)
    games = [g for g in a.games.split(",") if g] or None
    arms = [(t.split("=", 1)[0], load(t.split("=", 1)[1], games)) for t in a.arms]
    pairs = sorted(set.intersection(*(set(r) for _, r in arms)))
    md = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    if a.by_game:
        text = "\n".join(by_game(arms, pairs))
        print(text)
        if a.out:
            with open(a.out, "w") as f:
                f.write(text + "\n")
            print(f"\nwrote {a.out}")
        return
    lines = [f"{len(pairs)} pairs of (game, seed) that every arm has.", "",
             md(["Game", "Seed"] + [n for n, _ in arms] + ["Level-ups at action: " + " ; ".join(n for n, _ in arms)]),
             md(["---"] * (3 + len(arms)))]
    for g, s in pairs:
        cells = []
        for _, r in arms:
            run = r[g, s]
            who = [f"{run[k]} by the {k}" if k == "explorer" else f"{run[k]} by planning"
                   for k in ("planning", "explorer") if run[k]]
            cells.append(f"{len(run['ups'])}" + (f" ({', '.join(who)})" if who else ""))
        lines.append(md([g, s] + cells + [" ; ".join(" ".join(map(str, r[g, s]["ups"])) or "-" for _, r in arms)]))
    lines += ["", md(["Arm"] + [f"Levels by {t:,}" for t in BUDGETS] + ["By planning", "By the explorer", "Explorer's moves",
                             "Errors in the planner's files", "Against " + arms[0][0] + ": better / same / worse"]),
              md(["---"] * (6 + len(BUDGETS)))]
    base = arms[0][1]
    for name, r in arms:
        diff = [len(r[p]["ups"]) - len(base[p]["ups"]) for p in pairs]
        lines.append(md([name] + [sum(sum(u <= t for u in r[p]["ups"]) for p in pairs) for t in BUDGETS]
                        + [sum(r[p][k] for p in pairs) for k in ("planning", "explorer", "explorer_moves", "errors")]
                        + ["" if r is base else f"{sum(d > 0 for d in diff)} / {sum(d == 0 for d in diff)} / {sum(d < 0 for d in diff)}"]))
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
