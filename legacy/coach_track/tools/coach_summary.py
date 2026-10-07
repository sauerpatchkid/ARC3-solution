#!/usr/bin/env python3
"""coach_summary.py - print what the Coach's LLM would read at chosen moments
of a recorded run (docs/plans/llm-coach.md, step 3).

Replays a transition corpus through coach.history.History exactly as the live
agent would feed it, and renders the summary on the screen the next move is
made on.

  uv run python legacy/coach_track/tools/coach_summary.py <run>/transitions --at 20000
  uv run python legacy/coach_track/tools/coach_summary.py <run>/transitions --prewin      # before each winning move
  uv run python legacy/coach_track/tools/coach_summary.py <run>/transitions --stall 1500  # first stuck point per level

Prints each summary with its length and a token estimate, then the replay speed.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # legacy/coach_track
ROOT = os.path.dirname(os.path.dirname(TRACK))                               # the repo
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))   # upgrades.py, gridtools.py
sys.path.insert(0, TRACK)                                  # the coach package
from coach.history import History  # noqa: E402


def transitions(corpus):
    """(frame, action, next_frame, level, action_num) in order."""
    paths = sorted(glob.glob(os.path.join(corpus, "shard_*.npz")))
    if not paths:
        sys.exit(f"no shard_*.npz in {corpus}")
    for p in paths:
        with np.load(p) as z:
            f, a, nf, lv, an = z["frames"], z["actions"], z["next_frames"], z["levels"], z["action_nums"]
            for i in range(len(a)):
                yield f[i], int(a[i]), nf[i], int(lv[i]), int(an[i])


def winning_moves(corpus):
    """action_nums of level-completing moves: the transition before the level number rises."""
    out, prev = set(), None
    for _, _, _, lv, an in transitions(corpus):
        if prev is not None and lv > prev[0]:
            out.add(prev[1])
        prev = (lv, an)
    return out


def show(title, s):
    t = s["text"]
    print(f"\n===== {title} ({len(t)} chars, ~{len(t) / 2.8:.0f} tokens, {len(s['objects'])} objects) =====")
    print(t, end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus")
    ap.add_argument("--at", type=int, default=0, help="render before the move with this action_num")
    ap.add_argument("--prewin", action="store_true", help="render before every level-completing move")
    ap.add_argument("--stall", type=int, default=0, help="render at the first point per level with this many moves since a new screen")
    a = ap.parse_args()
    wins = winning_moves(a.corpus) if a.prewin else set()
    h, n, t0 = History(), 0, time.perf_counter()
    stalled = set()
    for frame, act, nxt, lv, an in transitions(a.corpus):
        if a.at and an >= a.at:
            show(f"before move {an}", h.render(frame))
            break
        if an in wins:
            show(f"before the winning move {an} (level {lv + 1})", h.render(frame))
        if a.stall and h.level is not None and lv == h.level.index and lv not in stalled \
                and h.level.moves - h.level.last_new >= a.stall:
            stalled.add(lv)
            show(f"stuck point at move {an} (level {lv + 1})", h.render(frame))
        h.observe(frame, act, nxt, lv)
        n += 1
    dt = time.perf_counter() - t0
    print(f"\nreplayed {n:,} transitions in {dt:.1f} s ({1e3 * dt / max(n, 1):.3f} ms each, rendering included)")


if __name__ == "__main__":
    main()
