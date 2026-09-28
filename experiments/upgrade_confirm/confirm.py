#!/usr/bin/env python3
"""confirm.py — the 25-game test of round 2's winner against the adopted agent.

Runs only the new version (mb_gated_att) on all 25 games x 3 seeds x 100k; the
novelty agent's runs are reused from the Plan B Confirm sweep. When the runs
finish, tools/paired_compare.py writes the verdict against the pre-registered
Confirm rule. Plan: docs/plans/upgrade-confirm.md.

    make upgrade-confirm DRY_RUN=1
    make upgrade-confirm
    make upgrade-confirm-status / upgrade-confirm-pause
    make upgrade-confirm RESUME=results/confirm_upgrade/<stamp>/manifest.tsv

Reuses experiments/upgrade_screen/screen.py as a library (parallel runs,
per-run results folders, pause, resume) without editing it.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "upgrade_screen"))

import benchmark  # noqa: E402
import screen  # noqa: E402

ARM = "mb_gated_att"
ENV = {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1", "EVAL_UPGRADES": "bars,map_gated,attempt"}
BASE = "results/sweeps/sweep_20260917_224431.manifest:A1"   # the novelty agent, Plan B Confirm

screen.ARMS = {ARM: (ENV, None, "round 2 winner: map + bars + walk back only while it pays + half credit per attempt")}
screen.GAMES = list(benchmark.ALL_GAMES)
screen.SEEDS = [0, 1, 2]
screen.CAP = 100_000
screen.results_root = lambda: os.path.join(os.getenv("EVAL_RESULTS_DIR", "results"), "confirm_upgrade")


def verdict(argv):
    manifest = argv[0]
    out = os.path.join(os.path.dirname(manifest), "verdict.md")
    subprocess.run([sys.executable, "tools/paired_compare.py", "--base", BASE,
                    "--new", f"{manifest}:{ARM}", "--out", out], cwd=ROOT)


if __name__ == "__main__":
    import rank
    rank.main = verdict            # the screen runner calls rank.main at the end
    screen.main()
