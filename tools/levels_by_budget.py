#!/usr/bin/env python3
"""levels_by_budget.py - two arms compared on WHEN levels were completed, not only how many.

The problem it fixes: every verdict so far counts levels at the end of a run
(100,000 actions). A guide that acts in the first moves of each new level
should show up as levels completed EARLIER, which that count cannot see.

    uv run python tools/levels_by_budget.py \\
        --base results/glassbox/g4/online/manifest.tsv:A0-fresh \\
        --new  results/glassbox/g4/online/manifest.tsv:GB \\
        [--budgets 2000,10000,30000,100000] [--games vc33,ft09] [--by-game] [--out table.md]

Each side is MANIFEST:ARM, as for tools/paired_compare.py, and only (game,
seed) pairs present on both sides are compared. It reads each run's
metrics.json (`levelup_events`: the action at which each level was completed,
on the actions-taken clock) and reports:

  1. Levels completed by each budget, summed over the pairs, with the paired
     better / same / worse count and a 95% bootstrap interval for the
     difference (games resampled, as paired_compare does). A run that stopped
     before a budget (a game won early) is counted there with the levels it
     had, and is named in the output.
  2. AULC (analyze_curves.py: the average completed level, weighted uniformly
     in log action budget), from 100 actions to the shortest run's budget.
  3. Level by level, which arm finished the level in fewer actions (see
     level_races). Level 1 is shown apart from levels 2 and up: nothing can be
     carried into level 1, so it shows what chance alone does. --by-game lists
     levels 2 and up per game.

It gives no verdict: no rule for these numbers was registered before the runs,
so every use of this table is exploratory and must be labelled so. To see how
large a difference chance alone makes, run it on two arms of the same agent
(for G4: A0's 27 Sep runs against the same-day A0).
"""
import argparse
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from analyze_curves import T_MIN_DEFAULT, aulc, level_at, read_metrics  # noqa: E402
from manifest import read_manifest  # noqa: E402
from paired_compare import bootstrap_delta  # noqa: E402

BUDGETS = (2000, 10000, 30000, 100000)
EARLY = 0.99        # a run this far short of the largest budget stopped early (older runs record 99,999 of 100,000)


def load(spec, games=None):
    """{(game, seed): (levelup_events, actions taken)} for one MANIFEST:ARM."""
    path, arm = spec.rsplit(":", 1)
    out = {}
    for row in read_manifest(path, strict=True):
        if row["arm"] != arm or (games and row["game"] not in games):
            continue
        d = row["run_dir"] if os.path.isabs(row["run_dir"]) else os.path.join(ROOT, row["run_dir"])
        m = read_metrics(d)
        out[(row["game"], row["seed"])] = (m.get("levelup_events") or [], m["_budget"])
    return out


def paired(base, new, value):
    """Totals, better/same/worse and the bootstrap interval of `value(run)` over the common pairs."""
    pairs = sorted(set(base) & set(new))
    b = {k: value(base[k]) for k in pairs}
    n = {k: value(new[k]) for k in pairs}
    eps = 1e-9
    wins = sum(n[k] > b[k] + eps for k in pairs)
    losses = sum(n[k] < b[k] - eps for k in pairs)
    per_game = {}
    for k in pairs:
        per_game[k[0]] = per_game.get(k[0], 0.0) + n[k] - b[k]
    lo, hi = bootstrap_delta(list(per_game.values()))
    return dict(base=sum(b.values()), new=sum(n.values()), wins=wins, ties=len(pairs) - wins - losses,
                losses=losses, lo=lo, hi=hi)


def level_races(base, new):
    """One row per (pair, level) that BOTH runs started and at least one finished:
    (game, seed, level, winner, ratio). The time on a level runs from the level-up
    before it (or the run's start) to its own level-up. A run that never finished a
    level spent at least the rest of its budget on it, so:

      both finished      the fewer actions wins; ratio = new / base
      one finished       it wins if it took fewer actions than the other spent on the
                         level without finishing; otherwise the race is "undecided"
                         (the other might still have been the faster); no ratio

    Leaving the unfinished levels out instead would favour whichever arm finished
    FEWER levels, because its slow levels are the ones that drop out."""
    rows = []
    for k in sorted(set(base) & set(new)):
        (events_b, cap_b), (events_n, cap_n) = base[k], new[k]
        ab, an = sorted(a for a, _ in events_b), sorted(a for a, _ in events_n)
        for i in range(min(len(ab), len(an)) + 1):           # both finished levels 1..i, so both started i+1
            start_b, start_n = (ab[i - 1] if i else 0), (an[i - 1] if i else 0)
            x = ab[i] - start_b if i < len(ab) else None      # None: never finished
            y = an[i] - start_n if i < len(an) else None
            if x is None and y is None:
                continue
            if x is not None and y is not None:
                winner = "new" if y < x else "base" if x < y else "same"
                rows.append((k[0], k[1], i + 1, winner, y / x if x > 0 else None))
            elif y is not None:
                rows.append((k[0], k[1], i + 1, "new" if y < cap_b - start_b else "undecided", None))
            else:
                rows.append((k[0], k[1], i + 1, "base" if x < cap_n - start_n else "undecided", None))
    return rows


def race_row(name, rows):
    new, base = sum(r[3] == "new" for r in rows), sum(r[3] == "base" for r in rows)
    per_game = {}
    for g, _, _, winner, _ in rows:
        per_game[g] = per_game.get(g, 0) + (winner == "new") - (winner == "base")
    lo, hi = bootstrap_delta(list(per_game.values())) if per_game else (0, 0)
    ratios = [r[4] for r in rows if r[4] is not None]
    median = f"{st.median(ratios):.2f} ({len(ratios)} levels)" if ratios else "-"
    return (f"| {name} | {len(rows)} | {len(per_game)} | {new} | {base} | {len(rows) - new - base} | "
            f"{new - base:+d} | [{lo:+.0f}, {hi:+.0f}] | {median} |")


def by_game_table(rows):
    """Levels 2 and up, game by game: is the pooled row one game's doing?"""
    lines = ["| Game | levels 2 and up | new faster | base faster | same or undecided |", "|---|---:|---:|---:|---:|"]
    for g in sorted({r[0] for r in rows}):
        mine = [r for r in rows if r[0] == g]
        new, base = sum(r[3] == "new" for r in mine), sum(r[3] == "base" for r in mine)
        lines.append(f"| {g} | {len(mine)} | {new} | {base} | {len(mine) - new - base} |")
    return lines


def report(base, new, budgets=BUDGETS, t_min=T_MIN_DEFAULT, by_game=False):
    pairs = sorted(set(base) & set(new))
    if not pairs:
        raise SystemExit("no (game, seed) pairs in common")
    shortest = min(min(base[k][1], new[k][1]) for k in pairs)
    early = [f"{name} {k[0]} seed {k[1]} ({side[k][1]:,} actions, {len(side[k][0])} levels)"
             for name, side in (("base", base), ("new", new)) for k in pairs
             if side[k][1] < EARLY * max(budgets)]
    lines = [f"Pairs compared: {len(pairs)} over {len({g for g, _ in pairs})} games. "
             f"Shortest run: {shortest:,} actions."]
    if early:                       # a game won before the cap, or a run cut short
        lines.append(f"Stopped before {max(budgets):,} actions, and counted at later budgets with the levels "
                     f"it had when it stopped: {'; '.join(early)}.")
    lines += ["",
              "| Levels completed by | base | new | new - base | better / same / worse | 95% interval (games resampled) |",
              "|---|---:|---:|---:|---|---|"]
    for t in budgets:
        r = paired(base, new, lambda run, t=t: level_at(run[0], t))
        lines.append(f"| {t:,} actions | {r['base']} | {r['new']} | {r['new'] - r['base']:+d} | "
                     f"{r['wins']} / {r['ties']} / {r['losses']} | [{r['lo']:+.0f}, {r['hi']:+.0f}] |")
    r = paired(base, new, lambda run: aulc(run[0], t_min, float(shortest)))
    lines += ["",
              f"AULC, {t_min:g} to {shortest:,} actions, summed over the pairs: base {r['base']:.2f}, "
              f"new {r['new']:.2f} ({r['new'] - r['base']:+.2f}); better / same / worse "
              f"{r['wins']} / {r['ties']} / {r['losses']}; 95% interval [{r['lo']:+.2f}, {r['hi']:+.2f}]."]
    races = level_races(base, new)
    later = [r for r in races if r[2] >= 2]
    lines += ["",
              "| Which arm finished a level in fewer actions | levels | games | new | base | same or undecided "
              "| new - base | 95% interval (games resampled) | median ratio new / base, both finished |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|",
              race_row("level 1", [r for r in races if r[2] == 1]),
              race_row("levels 2 and up", later)]
    if by_game and later:
        lines += [""] + by_game_table(later)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="MANIFEST:ARM for the reference arm")
    ap.add_argument("--new", required=True, help="MANIFEST:ARM for the candidate arm")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)), help="action budgets, comma-separated")
    ap.add_argument("--games", default="", help="only these games, comma-separated (default: all)")
    ap.add_argument("--by-game", action="store_true", help="also list levels 2 and up game by game")
    ap.add_argument("--out", default="", help="optional .md path for the table")
    a = ap.parse_args()
    games = set(a.games.split(",")) if a.games else None
    text = report(load(a.base, games), load(a.new, games), [int(x) for x in a.budgets.split(",")],
                  by_game=a.by_game)
    head = f"base: `{a.base}`  \nnew: `{a.new}`" + (f"  \ngames: {a.games}" if a.games else "")
    print(head.replace("  \n", "\n") + "\n\n" + text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"# Levels by action budget (exploratory)\n\n{head}\n\n{text}\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
