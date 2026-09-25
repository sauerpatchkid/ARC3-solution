#!/usr/bin/env python3
"""screen.py — the upgrade screen: every candidate improvement, short runs, ranked.

A SCREEN, not a test: 2 seeds x 50k actions on 8 games, to find which ideas
deserve a proper dev test (3 seeds x 100k, pre-registered rule). Candidates
and the reasoning behind each: docs/plans/upgrade-screen.md and
docs/reports/Goose_Upgrade_Options.docx.

    uv run python experiments/upgrade_screen/screen.py --dry-run      # plan + ETA
    make upgrade-screen                                                 # detached
    make upgrade-screen-status
    make upgrade-screen-pause          # finish the runs in progress, then stop
    make upgrade-screen RESUME=results/screen/<stamp>/manifest.tsv

Runs go in parallel (--jobs, default 4), but the GPU is shared, so parallel
runs barely add up. Measured on this machine (RTX 5090, 10k-action runs):
total speed 140 act/s with 1 run, 155 with 2, 162 with 4, 157 with 6. Four at
once is the best setting and gains ~15%, not 4x. Each run gets its OWN results directory
(results/screen/<stamp>/runs/<arm>/<game>_s<seed>/), because run directories
are named by the second they start and parallel runs would otherwise collide.
Every finished run is scored and appended to the manifest immediately, so a
pause, crash or sleep loses at most the runs in progress; --resume skips
everything already in the manifest. When all runs are done, rank.py writes
the leaderboard.

Everything here is removable: delete experiments/upgrade_screen/ and the
lines tagged [upgrades] (see custom_agents/upgrades.py).
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NOVEL = {"EVAL_LABEL": "novel"}
MAP = {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1"}

# name -> (environment, reference arm it modifies, one-line description)
ARMS = {
    "novelty":     (NOVEL, None, "reference: the adopted agent (novelty label)"),
    "map":         (MAP, None, "reference: novelty label + return map"),
    "bars":        (dict(NOVEL, EVAL_UPGRADES="bars"), "novelty", "catch progress bars that move with real changes"),
    "attempt":     (dict(NOVEL, EVAL_UPGRADES="bars,attempt"), "novelty", "half credit for screens new this attempt (with bars)"),
    "graded":      (dict(NOVEL, EVAL_UPGRADES="graded"), "novelty", "reward 1/sqrt(visits) instead of 1-or-0"),
    "deadclick":   (dict(NOVEL, EVAL_UPGRADES="deadclick"), "novelty", "stop clicking cells that did nothing 4 times"),
    "persist":     (dict(NOVEL, EVAL_RESET_ON_LEVEL="0"), "novelty", "keep the network when a level is completed"),
    "map_stuck":   (dict(MAP, EVAL_MAP_RETURN="0"), "map", "map only when stuck: no walk back after game over"),
    "map_gated":   (dict(MAP, EVAL_UPGRADES="map_gated"), "map", "walk back only while it pays (per-game bandit)"),
    "map_objects": (dict(MAP, EVAL_UPGRADES="map_objects"), "map", "object-level 'untried' on click screens"),
    "map_diverse": (dict(MAP, EVAL_UPGRADES="map_diverse"), "map", "Go-Explore target choice: less-visited first"),
    "map_bars":    (dict(MAP, EVAL_UPGRADES="bars"), "map", "map + novelty with progress bars masked"),
}
# Two games from each situation the candidates target: the map helped (tu93,
# su15), the map hurt (ar25, tr87), a missed progress bar makes the walk back
# drift (dc22, g50t), lots of dead clicks (vc33; su15 too), and the flagship.
GAMES = "tu93 su15 ar25 tr87 dc22 g50t vc33 ft09".split()
AGGREGATE_ACT_S = {1: 140, 2: 155, 3: 158, 4: 162, 5: 160, 6: 157}   # measured, see docstring
SEEDS = [0, 1]
CAP = 50_000
TRANSITIONS = re.compile(r"^\[run_local\] transitions: (.+)$", re.M)


def results_root():
    return os.path.join(os.getenv("EVAL_RESULTS_DIR", "results"), "screen")


def read_manifest(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                done.add((parts[1], parts[2], parts[3]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--resume", default="", help="an existing manifest.tsv to continue")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    games = [g for g in a.games.split(",") if g]
    seeds = [s for s in a.seeds.split(",") if s]
    arms = [x for x in a.arms.split(",") if x]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        sys.exit(f"unknown arms {bad}; known {list(ARMS)}")

    root = results_root()
    if a.resume:
        manifest = a.resume
        out = os.path.dirname(manifest)
    else:
        out = os.path.join(root, datetime.now().strftime("%Y%m%d_%H%M%S"))
        manifest = os.path.join(out, "manifest.tsv")
    done = read_manifest(manifest)
    # seeds outer, games, arms inner: a partial screen is still paired
    jobs = [(g, s, arm) for s in seeds for g in games for arm in arms if (g, s, arm) not in done]

    speed = AGGREGATE_ACT_S.get(a.jobs, 155)      # total act/s across parallel runs
    eta_h = (len(jobs) * a.cap / speed + len(jobs) * 45 / max(a.jobs, 1)) / 3600
    print("=" * 70)
    print(f" UPGRADE SCREEN  {len(arms)} arms x {len(games)} games x {len(seeds)} seeds x {a.cap} actions")
    print(f" runs to do: {len(jobs)} (already done: {len(done)})   parallel jobs: {a.jobs}")
    print(f" ETA: ~{eta_h:.1f} h (measured total speed ~{speed} act/s with {a.jobs} at once)")
    print(f" output: {out}")
    print("=" * 70)
    for arm in arms:
        env, ref, what = ARMS[arm]
        print(f"   {arm:<12} vs {ref or '-':<8} {what}")
    if a.dry_run:
        print("DRY RUN - nothing executed.")
        return

    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    stop_file = os.path.join(root, "STOP")
    pid_file = os.path.join(root, "screen.pid")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    lock = threading.Lock()

    def run(job):
        g, s, arm = job
        env = dict(os.environ)
        env.update(ARMS[arm][0])
        env.update(EVAL_SEED=str(s), EVAL_MAX_ACTIONS=str(a.cap), PYTHONHASHSEED="0",
                   EVAL_RESULTS_DIR=os.path.join(out, "runs", arm, f"{g}_s{s}"))
        log = os.path.join(out, "logs", f"{arm}_{g}_s{s}.log")
        t0 = time.time()
        with open(log, "w") as f:
            subprocess.run([sys.executable, "run_local.py", "--game", g], cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
        text = open(log).read()
        m = TRANSITIONS.search(text)
        if not m:
            print(f"!! {arm} {g} s{s}: no corpus (see {log})", flush=True)
            return
        corpus = m.group(1).strip()
        subprocess.run([sys.executable, "compute_metrics.py", corpus, "--game", g,
                        "--agent", f"screen_{arm}", "--seed", str(s)],
                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        rundir = os.path.dirname(corpus)
        if not os.path.exists(os.path.join(ROOT, rundir, "metrics.json")) and \
                not os.path.exists(os.path.join(rundir, "metrics.json")):
            print(f"!! {arm} {g} s{s}: scoring failed", flush=True)
            return
        levels = len(re.findall(r"Score changed from [0-9]", text))
        with lock:
            with open(manifest, "a") as f:
                f.write(f"{rundir}\t{g}\t{s}\t{arm}\n")
            n_done = len(read_manifest(manifest))
        print(f"done {arm:<12} {g} s{s}  levels={levels}  {time.time() - t0:5.0f}s  "
              f"({n_done} in manifest)", flush=True)

    running = []
    stopped = False
    try:
        for job in jobs:
            while len([t for t in running if t.is_alive()]) >= a.jobs:
                time.sleep(2)
            if os.path.exists(stop_file):
                stopped = True
                break
            t = threading.Thread(target=run, args=(job,), daemon=False)
            t.start()
            running.append(t)
            print(f"start {job[2]:<12} {job[0]} s{job[1]}  {datetime.now():%H:%M:%S}", flush=True)
            time.sleep(1.5)                       # stagger GPU start-up
        for t in running:
            t.join()
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)
    if stopped or os.path.exists(stop_file):
        if os.path.exists(stop_file):
            os.remove(stop_file)
        print(f"=== PAUSED. Resume with: make upgrade-screen RESUME={manifest}", flush=True)
        return
    missing = [j for j in jobs if j not in read_manifest(manifest)]
    import rank
    rank.main([manifest])
    if missing:
        print(f"=== FINISHED WITH {len(missing)} FAILED RUNS (see '!!' lines and logs/). "
              f"Retry them with: make upgrade-screen RESUME={manifest}", flush=True)
    else:
        print(f"=== DONE. Leaderboard: {os.path.join(out, 'leaderboard.md')}", flush=True)


if __name__ == "__main__":
    main()
