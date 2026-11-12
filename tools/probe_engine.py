#!/usr/bin/env python3
"""probe_engine.py - what does the local engine give that Goose does not use?

The problem it fixes: Goose reads only the LAST frame the engine returns for a
move, and never sends ACTION7 (run_local.py drops it: the harness enum cannot
name it). Whether that loses anything was never measured. This plays each game
with uniformly random moves, straight on the engine (no agent, no GPU), and
counts:

  FRAMES   how many frames a move returns. Then, for the moves Goose reads as
           "nothing happened" (the last frame equals the screen before the
           move): how many showed something else in between. And for the moves
           that did change the screen: how many passed through a screen that is
           neither the one before nor the one after
  ACTION7  on the games that offer it: how often it changes the screen, and how
           often that change puts the screen back as it was before the previous
           move (the engine's documented meaning: undo), exactly or but for a
           few cells (a counter that ticked)
  SCREENS  distinct screens reached with ACTION7 in the mix and without it,
           counted as the agent counts them: under its own decoration mask
           (canon.OnlineCanonicalizer and the `bars` upgrade's BarDetector), so
           a counter that ticks on every move does not make every screen new

    uv run python tools/probe_engine.py [--games sb26,lf52] [--steps 2000] [--seed 0] [--out table.md]

Random play only shows what the engine CAN return. It does not say an agent
would gain from it: that takes a run of the agent.
"""
import argparse
import logging
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from canon import OnlineCanonicalizer  # noqa: E402
from upgrades import BarDetector  # noqa: E402

GRID = 64
FEW = 8             # "back as it was, but for a few cells" (a counter or bar that ticked)


def probe(env, steps, rng, allow7=True):
    """Play `steps` random moves on `env`. Returns the counts (module doc)."""
    obs = env.reset()
    last = np.array(obs.frame)[-1]
    before_prev = None              # the screen before the previous move, if that move changed it
    c = dict(moves=0, multi=0, max_frames=1, still=0, still_flash=0, changing=0, changing_between=0,
             a7=0, a7_changed=0, a7_after_change=0, a7_undid=0, a7_undid_nearly=0,
             game_overs=0, levels=0, offered=set())
    canon, bars, seen = OnlineCanonicalizer(), BarDetector(), [last]
    while c["moves"] < steps:
        offered = [int(a) for a in (obs.available_actions or [])]
        c["offered"].update(offered)
        if obs.state.name in ("GAME_OVER", "WIN", "NOT_PLAYED") or not offered:
            c["game_overs"] += obs.state.name == "GAME_OVER"
            obs = env.reset()
            last, before_prev = np.array(obs.frame)[-1], None
            continue
        choices = [a for a in offered if a != 0 and (allow7 or a != 7)]
        a = rng.choice(choices)
        action = getattr(type(env.action_space[0]), f"ACTION{a}")      # the engine's enum, by name
        data = {"x": rng.randrange(GRID), "y": rng.randrange(GRID)} if a == 6 else None
        level = obs.levels_completed
        obs = env.step(action, data=data)
        frames = np.array(obs.frame)
        c["moves"] += 1
        c["levels"] += obs.levels_completed > level
        cur = frames[-1]
        changed = bool((cur != last).any())
        # a frame before the last one that is neither the screen before the move nor the screen after it
        between = any((f != last).any() and (f != cur).any() for f in frames[:-1])
        c["multi"] += len(frames) > 1
        c["max_frames"] = max(c["max_frames"], len(frames))
        c["changing" if changed else "still"] += 1
        c["changing_between" if changed else "still_flash"] += between
        if a == 7:
            c["a7"] += 1
            c["a7_changed"] += changed
            if before_prev is not None:                 # the previous move changed the screen: could it be undone?
                c["a7_after_change"] += 1
                left = int((cur != before_prev).sum())  # cells still different from the screen two moves back
                c["a7_undid"] += changed and left == 0
                c["a7_undid_nearly"] += changed and 0 < left <= FEW and left < int((last != before_prev).sum())
        canon.update(last, cur)
        bars.update(last, cur)
        before_prev = last if changed and a != 7 else None
        last = cur
        seen.append(cur)
    live = ~(canon.mask | bars.sticky)                  # the agent's mask at the end of the probe
    c["screens"] = len({(f * live).tobytes() for f in seen})
    return c


def pct(n, d):
    return f"{n / d:.0%}" if d else "-"


def row(game, with7, without7):
    c = with7
    offered = " ".join(str(a) for a in sorted(c["offered"]))
    frames = f"{pct(c['multi'], c['moves'])} (up to {c['max_frames']})" if c["multi"] else "never"
    flash = f"{pct(c['still_flash'], c['still'])} of {c['still']:,}"
    between = f"{pct(c['changing_between'], c['changing'])} of {c['changing']:,}"
    if c["a7"]:
        a7 = (f"{pct(c['a7_changed'], c['a7'])} | {pct(c['a7_undid'], c['a7_after_change'])} exactly, "
              f"{pct(c['a7_undid_nearly'], c['a7_after_change'])} but for a few cells | "
              f"{c['screens']:,} / {without7['screens']:,}")
    else:
        a7 = f"- | - | {c['screens']:,}"
    return f"| {game} | {offered} | {frames} | {flash} | {between} | {a7} |"


HEADER = ["| Game | Actions offered | Moves returning several frames | \"Nothing happened\" moves that showed "
          "something in between | Changing moves that passed through another screen | ACTION7 changed the screen "
          "| ... and undid the previous move | Distinct screens: with / without ACTION7 |",
          "|---|---|---|---|---|---|---|---|"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", default="", help="comma-separated (default: all 25)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="optional .md path for the table")
    a = ap.parse_args()
    logging.disable(logging.CRITICAL)            # the engine logs every game it loads
    import benchmark
    import run_local
    games = a.games.split(",") if a.games else benchmark.ALL_GAMES
    arc = run_local.make_arcade(offline=True)
    lines = [f"{a.steps:,} uniformly random moves per game, seed {a.seed}, on the local engine. Actions are the "
             f"engine's numbers (1-5 buttons, 6 click, 7 ACTION7).", ""] + HEADER
    for game in games:
        runs = []
        for allow7 in (True, False):
            env = run_local.make_env(arc, game, a.seed, None)
            runs.append(probe(env, a.steps, random.Random(a.seed), allow7))
            if 7 not in runs[0]["offered"]:
                runs.append(runs[0])             # no ACTION7 here: one run is enough
                break
        lines.append(row(game, *runs))
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"# Engine probe\n\n{text}\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
