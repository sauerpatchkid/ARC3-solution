#!/usr/bin/env python3
"""gb_offline.py - Glassbox's offline driver: one command per stage, files in, files out.

Each stage's code is its own file in custom_agents/gb/; this only finds the
inputs, opens the run folder (gb/artifacts.py) and calls the stage. Stages are
added here as they are built (plan section 4.3).

    uv run python tools/gb_offline.py --import-cp1 [--games ft09,dc22] [--run cp1_import]
    uv run python tools/gb_offline.py --stage mask --run <new> --from cp1_import
    uv run python tools/gb_offline.py --stage see|check|carry|guards|show --run <name> [--games a,b]

--import-cp1  re-run Rulebook v2's CP1 with gb/core (no LLM, no search) and write
              it as Glassbox artifacts; every number is checked against CP1's
              registered one (custom_agents/gb/cp1_import.py). Exit status 1 if
              any differs.
--stage mask  S0 MASK (custom_agents/gb/s0_mask.py): rebuild each game's evidence
              from CP1's recordings and levels with the bars in the decoration
              mask run to the screen's edges; --from <run> copies that run's
              candidates and groups (S2's files) into the new run.
--stage see   S1 SEE: objects, what happened to them, closure (custom_agents/gb/s1_see.py),
              for every evidence split of each game in the run.
--stage check S3 CHECK's grading (custom_agents/gb/s3_check.py): every candidate
              replayed on the training level, and the book composed.
--stage carry S4 CARRY (custom_agents/gb/s4_carry.py): the book's rules re-bound on
              the transfer level's fit split; T0 and T1 on its score split.
--stage guards  S3's guard fitter (custom_agents/gb/s3_guards.py): every
              admitted-but-not-trusted rule, a guard from S1's facts, the
              narrowed rule and the exception, re-graded; the book and T1
              before and after. On CP1's candidates it needs --g0: allowed only
              in a run registered beforehand (G0: prereg section 3; G0b: 3b).
--stage show  S6 SHOW: the run's report page, funnel figure and case viewer
              (custom_agents/gb/s6_show.py) in results/glassbox/<run>/s6_show/.
--config      another settings file than experiments/glassbox/glassbox.toml.
--set S.K=V   override one setting for this command (repeatable), e.g. --set
              mask.complete_bars=false; stages record the settings they used.
--compare-cp1 does --run (made by --stage mask, check, carry) give CP1's results?
              Exit status 1 if not (cp1_import.matches).
"""
import argparse
import json
import os
import shutil
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import artifacts as A  # noqa: E402
from gb import cp1_import  # noqa: E402
from gb import s0_mask, s1_see, s3_check, s3_guards, s4_carry  # noqa: E402
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


def seed_from(run, src, game):
    """S0's inputs for one game: CP1's recording and levels, and the source
    run's S2 files (candidates, LLM use, groups called) copied into `run`."""
    res = cp1_import.load_cp1(game)
    src_s2, dst_s2 = os.path.join(src.dir, "s2_write", game), os.path.join(run.stage_dir("s2_write"), game)
    os.makedirs(dst_s2, exist_ok=True)
    for f in ("candidates.jsonl", "llm.json", "groups.json"):
        shutil.copyfile(os.path.join(src_s2, f), os.path.join(dst_s2, f))
    return res["corpus"], res["train"]["level"], res.get("transfer_level")


def games_of(run):
    """The games a run has candidates for."""
    d = os.path.join(run.dir, "s2_write")
    return sorted(g for g in os.listdir(d) if not g.startswith("_") and os.path.isdir(os.path.join(d, g))) \
        if os.path.isdir(d) else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--import-cp1", action="store_true", help="re-run CP1 with gb/core into Glassbox artifacts")
    ap.add_argument("--games", default="", help="comma-separated (default: every game in the run)")
    ap.add_argument("--run", default="cp1_import", help="run folder name under results/glassbox/")
    ap.add_argument("--cp1", default=cp1_import.CP1_DIR, help="CP1's registered result folder (read only)")
    ap.add_argument("--workers", type=int, default=cp1_import.WORKERS)
    ap.add_argument("--stage", choices=["mask", "see", "check", "carry", "guards", "show"],
                    help="run one stage on --run (more are added as they are built)")
    ap.add_argument("--from", dest="src", default="cp1_import", help="--stage mask: the run to take candidates from")
    ap.add_argument("--config", default=A.DEFAULT_CONFIG, help="the settings file")
    ap.add_argument("--set", action="append", default=[], help="override one setting: section.key=value (TOML value)")
    ap.add_argument("--compare-cp1", action="store_true", help="does --run give CP1's results? (cp1_import.matches)")
    ap.add_argument("--force", action="store_true", help="re-run a stage even if its files are current")
    ap.add_argument("--g0", action="store_true",
                    help="allow --stage guards on CP1's candidates: only in a registered run (G0, G0b)")
    a = ap.parse_args()
    if a.import_cp1:
        sys.exit(import_cp1(a))
    if not a.stage:
        if not a.compare_cp1:
            ap.error("nothing to do: give --import-cp1, --stage or --compare-cp1")
    log = lambda m: print(m, flush=True)
    if a.stage == "show":
        s6_show.build(a.run, log=log)
        return
    run = A.Run(a.run, config_path=a.config)
    for item in a.set:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        run.config.setdefault(section, {})[name] = tomllib.loads(f"v = {value}")["v"]
    if a.compare_cp1:
        bad = []
        for game in (a.games.split(",") if a.games else games_of(run)):
            got = cp1_import.matches(run, game, a.cp1)
            bad += [game] if not all(got.values()) else []
            print(f"[{game}] " + ", ".join(f"{k} {'equal' if v else 'DIFFER'}" for k, v in got.items()), flush=True)
        print("ALL EQUAL TO CP1" if not bad else f"DIFFER on {', '.join(bad)}")
        sys.exit(1 if bad else 0)
    if a.stage == "mask":
        src = A.Run(a.src)
        for game in (a.games.split(",") if a.games else games_of(src)):
            s0_mask.run_game(run, game, *seed_from(run, src, game), log=log, force=a.force)
        return
    stage = {"see": s1_see.run_game, "check": s3_check.run_game, "carry": s4_carry.run_game,
             "guards": lambda run, game, log, force: s3_guards.run_game(run, game, g0=a.g0, log=log, force=force)}
    for game in (a.games.split(",") if a.games else games_of(run)):
        stage[a.stage](run, game, log=log, force=a.force)


if __name__ == "__main__":
    main()
