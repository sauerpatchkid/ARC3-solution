#!/usr/bin/env python3
"""frozen_moves.py - how long did a run sit on a screen that no move changed?

The problem it fixes: Goose sends RESET only after a GAME_OVER. On a screen that
no move changes, and that never ends in a game over, it stays until the run
ends. A watchdog ("after N moves with no change, send RESET") could give those
moves back, but nothing measured how many there are. This counts them in
finished recordings, so the watchdog's ceiling is known before it is built.

    uv run python tools/frozen_moves.py --runs results/glassbox/g4/online/manifest.tsv:A0-fresh \\
        [--after 300] [--jobs 8] [--out table.md]

A move is FROZEN when no live cell changed: the cells outside the agent's own
decoration mask, rebuilt move by move from the recording exactly as
mb_gated_att builds it (canon.OnlineCanonicalizer, then the `bars` upgrade's
BarDetector). A STRETCH is a run of frozen moves in a row. It ends at a move
that changes something, at a reset (the level restarted after a game over) and
at a level-up.

Per run it reports the longest stretch, the stretch the run ended in, and the
moves a watchdog firing after `--after` frozen moves could have cut: the part of
every stretch past its first `--after` moves. That is a ceiling, not a gain: a
reset only helps if the level's first screen leads somewhere new.
"""
import argparse
import glob
import multiprocessing as mp
import os
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from canon import OnlineCanonicalizer  # noqa: E402
from manifest import read_manifest  # noqa: E402
from upgrades import BarDetector  # noqa: E402


class Stretches:
    """The frozen stretches of one run, fed one recorded move at a time (module doc)."""

    def __init__(self):
        self.canon, self.bars = OnlineCanonicalizer(), BarDetector()
        self.done, self.run, self.moves = [], 0, 0          # finished stretches; the one in progress
        self.last_num = self.last_level = None

    def move(self, frame, next_frame, action_num, level):
        if self.run and (action_num != self.last_num + 1 or level != self.last_level):
            self.done.append(self.run)                       # a reset or a level-up ended the stretch
            self.run = 0
        self.canon.update(frame, next_frame)
        self.bars.update(frame, next_frame)
        if ((frame != next_frame) & ~(self.canon.mask | self.bars.sticky)).any():
            if self.run:
                self.done.append(self.run)
            self.run = 0
        else:
            self.run += 1
        self.last_num, self.last_level, self.moves = int(action_num), int(level), self.moves + 1

    def summary(self, after):
        every = self.done + ([self.run] if self.run else [])
        return dict(moves=self.moves, longest=max(every, default=0), tail=self.run,
                    cut=sum(max(0, n - after) for n in every), stretches_over=sum(n > after for n in every))


def scan(corpus, after):
    """One recording: {moves, longest, tail, cut, stretches_over}. `tail` is the stretch
    the run ended in (0 if its last move changed something)."""
    s = Stretches()
    for p in sorted(glob.glob(os.path.join(corpus, "shard_*.npz"))):
        with np.load(p) as z:
            f, n, nums, levels = z["frames"], z["next_frames"], z["action_nums"], z["levels"]
        for i in range(len(f)):
            s.move(f[i], n[i], nums[i], levels[i])
    return s.summary(after)


def _job(args):
    run_dir, game, seed, after = args
    d = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
    return game, seed, scan(os.path.join(d, "transitions"), after)


def table(rows, after):
    """Markdown: one line per game (seeds side by side), then the totals."""
    games = sorted({g for g, _, _ in rows})
    share = lambda r: r["cut"] / r["moves"] if r["moves"] else 0.0
    lines = [f"A watchdog firing after {after:,} frozen moves. `cut` = moves past the first {after:,} of every "
             f"frozen stretch, as a share of the run's recorded moves.", "",
             "| Game | cut, per seed | longest stretch, per seed | run ended frozen for (moves), per seed |",
             "|---|---|---|---|"]
    for g in games:
        mine = [r for gg, s, r in sorted(rows, key=lambda x: (x[0], int(x[1]))) if gg == g]
        lines.append(f"| {g} | {', '.join(f'{share(r):.0%}' for r in mine)} | "
                     f"{', '.join(f'{r['longest']:,}' for r in mine)} | {', '.join(f'{r['tail']:,}' for r in mine)} |")
    every = [r for _, _, r in rows]
    total = sum(r["moves"] for r in every)
    lines += ["",
              f"Runs: {len(every)}. Moves a watchdog could have cut: {sum(r['cut'] for r in every):,} of {total:,} "
              f"({sum(r['cut'] for r in every) / total:.1%}). Median run: {st.median(share(r) for r in every):.1%}.",
              "Runs where it could have cut at least 10% / 25% / 50% of the moves: "
              + " / ".join(str(sum(share(r) >= x for r in every)) for x in (0.10, 0.25, 0.50)) + ".",
              f"Runs that ended inside a frozen stretch longer than {after:,} moves: "
              f"{sum(r['tail'] > after for r in every)}."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="MANIFEST:ARM")
    ap.add_argument("--after", type=int, default=300, help="frozen moves before the watchdog fires")
    ap.add_argument("--games", default="", help="only these games, comma-separated")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="", help="optional .md path for the table")
    a = ap.parse_args()
    path, arm = a.runs.rsplit(":", 1)
    games = set(a.games.split(",")) if a.games else None
    jobs = [(r["run_dir"], r["game"], r["seed"], a.after) for r in read_manifest(path, strict=True)
            if r["arm"] == arm and (not games or r["game"] in games)]
    if not jobs:
        sys.exit(f"no runs of arm {arm} in {path}")
    with mp.Pool(a.jobs) as pool:
        rows = pool.map(_job, jobs)
    text = table(rows, a.after)
    print(f"runs: `{a.runs}`\n\n{text}")
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"# Frozen moves\n\nruns: `{a.runs}`\n\n{text}\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
