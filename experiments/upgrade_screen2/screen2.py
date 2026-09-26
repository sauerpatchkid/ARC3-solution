#!/usr/bin/env python3
"""screen2.py — upgrade screen, round 2: combinations of round 1's best ideas.

SEPARATE from round 1. It imports round 1's runner (experiments/upgrade_screen/
screen.py) and only swaps in its own arms, games and results folder; it never
edits a round-1 file. Round 1's best result is saved at git tag
`upgrade-screen-round1` and in results/screen/20260925_121259/.

No new agent code: every arm is a combination of switches that already exist
(EVAL_RETURN_MAP, EVAL_UPGRADES options in custom_agents/upgrades.py).

    uv run python experiments/upgrade_screen2/screen2.py --dry-run
    make upgrade-screen2                 # detached
    make upgrade-screen2-status
    make upgrade-screen2-pause
    make upgrade-screen2 RESUME=results/screen2/<stamp>/manifest.tsv

Design, arms and the pre-registered rule: docs/plans/upgrade-screen2.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "upgrade_screen"))

import screen  # noqa: E402  (round 1's runner, used as a library)

NOVEL = {"EVAL_LABEL": "novel"}
MAP = {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1"}


def up(*names):
    return dict(MAP, EVAL_UPGRADES=",".join(names))


# name -> (environment, reference, one-line description). Filled in from
# round 1's results; see docs/plans/upgrade-screen2.md for why each is here.
ARMS = {
    "novelty":        (NOVEL, None, "reference: the adopted agent (novelty label)"),
    "map_bars":       (up("bars"), None, "reference: round 1's best, unchanged"),
    "map_gated":      (up("map_gated"), "map_bars", "round 1's runner-up, unchanged"),
    "mb_gated":       (up("bars", "map_gated"), "map_bars", "bars + walk back only while it pays"),
    "mb_gated_att":   (up("bars", "map_gated", "attempt"), "map_bars", "... + half credit within an attempt"),
    "mb_gated_dead":  (up("bars", "map_gated", "deadclick"), "map_bars", "... + skip dead clicks"),
    "mb_all":         (up("bars", "map_gated", "attempt", "deadclick"), "map_bars", "bars + gated + attempt + dead clicks"),
    "mb_att":         (up("bars", "attempt"), "map_bars", "bars + half credit within an attempt"),
}
# Round 1's 8 games plus 3 it never used, to check the winners carry over.
GAMES = "tu93 su15 ar25 tr87 dc22 g50t vc33 ft09 sp80 cn04 bp35".split()


def results_root():
    return os.path.join(os.getenv("EVAL_RESULTS_DIR", "results"), "screen2")


screen.ARMS = ARMS
screen.GAMES = GAMES
screen.results_root = results_root

if __name__ == "__main__":
    # rank.py (round 1's) reads screen.ARMS, so it describes these arms; the
    # round-2 comparisons against map_bars are added by rank2.py.
    import rank
    import rank2
    rank.main = rank2.main
    screen.main()
