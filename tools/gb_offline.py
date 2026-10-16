#!/usr/bin/env python3
"""gb_offline.py - Glassbox's offline driver: one command per stage, files in, files out.

Each stage's code is its own file in custom_agents/gb/; this only finds the
inputs, opens the run folder (gb/artifacts.py) and calls the stage. Stages are
added here as they are built (plan section 4.3).

    uv run python tools/gb_offline.py --import-cp1 [--games ft09,dc22] [--run cp1_import]
    uv run python tools/gb_offline.py --stage see --run cp1_import [--games tu93,m0r0]
    uv run python tools/gb_offline.py --stage guards --run <name> [--games m0r0] [--g0]
    uv run python tools/gb_offline.py --stage show --run cp1_import

--import-cp1  re-run Rulebook v2's CP1 with gb/core (no LLM, no search) and write
              it as Glassbox artifacts; every number is checked against CP1's
              registered one (custom_agents/gb/cp1_import.py). Exit status 1 if
              any differs.
--stage see   S1 SEE: objects, what happened to them, closure (custom_agents/gb/s1_see.py),
              for every evidence split of each game in the run.
--stage guards  S3's guard fitter (custom_agents/gb/s3_guards.py): every
              admitted-but-not-trusted rule, a guard from S1's facts, the
              narrowed rule and the exception, re-graded; the book and T1
              before and after. On CP1's candidates this IS G0, so it needs --g0,
              which is for after prereg section 3 is registered.
--stage show  S6 SHOW: the run's report page, funnel figure and case viewer
              (custom_agents/gb/s6_show.py) in results/glassbox/<run>/s6_show/.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import artifacts as A  # noqa: E402
from gb import cp1_import  # noqa: E402
from gb import s1_see  # noqa: E402
from gb import s3_guards  # noqa: E402
from gb import s6_show  # noqa: E402


def import_cp1(a):
    games = a.games.split(",") if a.games else cp1_import.cp1_games(a.cp1)
    run = A.Run(a.run)
    print(f"importing CP1 from {os.path.relpath(a.cp1, ROOT)} into {os.path.relpath(run.dir, ROOT)}: "
          f"{', '.join(games)}", flush=True)
    summaries = []
    for game in games:
        t0 = time.time()
        check = cp1_import.import_game(game, run, a.cp1, a.workers, log=lambda m: print(m, flush=True))
        summaries.append(dict(check.summary(), seconds=round(time.time() - t0)))
    with open(os.path.join(run.dir, "cp1_check", "summary.json"), "w") as f:
        json.dump(summaries, f, indent=1)
    bad = [s["game"] for s in summaries if not s["ok"]]
    total = sum(s["checked"] for s in summaries)
    print(f"\n{total - sum(len(s['mismatches']) for s in summaries)} of {total} recomputed numbers equal CP1's "
          f"on {len(games)} games" + (f"; MISMATCH on {', '.join(bad)}" if bad else "; ALL MATCH"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--import-cp1", action="store_true", help="re-run CP1 with gb/core into Glassbox artifacts")
    ap.add_argument("--games", default="", help="comma-separated (default: every game CP1 ran)")
    ap.add_argument("--run", default="cp1_import", help="run folder name under results/glassbox/")
    ap.add_argument("--cp1", default=cp1_import.CP1_DIR, help="CP1's registered result folder (read only)")
    ap.add_argument("--workers", type=int, default=cp1_import.WORKERS)
    ap.add_argument("--stage", choices=["see", "guards", "show"],
                    help="run one stage on --run (more are added as they are built)")
    ap.add_argument("--g0", action="store_true",
                    help="allow --stage guards on CP1's candidates: that run IS G0 (prereg section 3 first)")
    ap.add_argument("--force", action="store_true", help="re-run a stage even if its files are current")
    a = ap.parse_args()
    if a.import_cp1:
        sys.exit(import_cp1(a))
    if a.stage == "see":
        run = A.Run(a.run)
        games = a.games.split(",") if a.games else s6_show.games_in(run.dir)
        for game in games:
            s1_see.run_game(run, game, log=lambda m: print(m, flush=True), force=a.force)
        return
    if a.stage == "guards":
        run = A.Run(a.run)
        games = a.games.split(",") if a.games else s6_show.games_in(run.dir)
        for game in games:
            s3_guards.run_game(run, game, g0=a.g0, log=lambda m: print(m, flush=True), force=a.force)
        return
    if a.stage == "show":
        s6_show.build(a.run, log=lambda m: print(m, flush=True))
        return
    ap.error("nothing to do: give --import-cp1 (stage commands are added as each stage is built)")


if __name__ == "__main__":
    main()
