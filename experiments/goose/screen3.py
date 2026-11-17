#!/usr/bin/env python3
"""screen3.py - upgrade screen 3: three Goose upgrades, alone and combined, at a short budget.

The problem it fixes: 9 arms x 25 games x 5,000 actions is 225 runs and several
hours of GPU time with nobody watching. This starts them detached, 4 at a time,
says how far they are, pauses and resumes them, retries a failed run once, and
writes the tables when the last run is in. The upgrades, the arms and the rule
the tables are read by: docs/1-goose-upgrades/upgrade-screen3.md.

    uv run python experiments/goose/screen3.py plan      the arms, their flags, the run count, a time estimate
    uv run python experiments/goose/screen3.py smoke     the GPU checks before the sweep (results/screen3_smoke/)
    uv run python experiments/goose/screen3.py start     begin, detached (it outlives the terminal)
    uv run python experiments/goose/screen3.py status    runs done, runs in progress, time left
    uv run python experiments/goose/screen3.py pause     no new run starts; the runs in progress finish, then it stops
    uv run python experiments/goose/screen3.py resume    continue after a pause, a crash or a reboot
    uv run python experiments/goose/screen3.py report    the tables (experiments/goose/screen3_report.py); also on a
                                                         partial manifest, and it says what is missing

The run set is fixed here (ARMS, benchmark.ALL_GAMES, seed 0, 5,000 actions):
`start` takes no arm, game or budget. Every run is `run_local.py --agent goose`
with its arm's flags; no preset is used, because a preset refuses other flags.

Output: results/screen3/<stamp>/ (results/screen3/CURRENT names the sweep the
commands act on). Each run has its own EVAL_RESULTS_DIR under runs/<arm>/, and
enters manifest.tsv (run_dir <TAB> game <TAB> seed <TAB> arm) only after it has
been played to its cap, or to the end of the game, and scored. So a pause loses
nothing, and a crash or a reboot loses only the runs in progress. A run that
fails is written to failed.tsv and skipped; when the queue is empty the failed
runs are tried once more, and then the report is written.

Games are the outer loop, with the four tiers of tiers.py taken in turn, so a
sweep that is cut short still covers every tier and is still paired by game.

TRIAL=1 does all of this on a tiny set (2 games x 3 arms x 300 actions, under
results/screen3_trial/), to try start, pause and resume before the real sweep.

Modelled on legacy/goose_experiments/upgrade_screen/screen.py (the runner) and
experiments/goose/carry_test.sh (the commands); it imports neither.
"""
import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import threading
import time
from collections import namedtuple
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "tools"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark  # noqa: E402
import manifest as _manifest  # noqa: E402  (the repo's one manifest reader)
import tiers  # noqa: E402

# ---- the run set ---------------------------------------------------------------------------------
SEED = 0
CAP = 5000
JOBS = 4                    # runs at a time (the GPU is shared: 4 is the measured best)
OFFLINE = True              # every run uses run_local.py --offline (what cpu_check uses; no network needed)
RUN_TIMEOUT = 3600          # seconds; a run that takes longer is killed and counted as failed
COMMON = {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1", "EVAL_SEED": str(SEED), "PYTHONHASHSEED": "0"}

GATED = "bars,map_gated,attempt"            # the adopted Goose's three upgrades (A0)
SCHED = "bars,map_sched,attempt"            # S: map_sched takes map_gated's place
TRAIN_EVERY = 1                             # the T arms (the plan page says when this may be 2)
TRAIN = {"EVAL_TRAIN_EVERY": str(TRAIN_EVERY), "EVAL_TRAIN_RECENT": "16"}

# arm -> (EVAL_UPGRADES, training flags, the upgrades it has), in the order the arms run within a game
ARMS = {
    "base": (GATED, {}, ""),
    "S": (SCHED, {}, "S"),
    "O": (GATED + ",objclick", {}, "O"),
    "T": (GATED, TRAIN, "T"),
    "SOT": (SCHED + ",objclick", TRAIN, "SOT"),
    "SO": (SCHED + ",objclick", {}, "SO"),
    "ST": (SCHED, TRAIN, "ST"),
    "OT": (GATED + ",objclick", TRAIN, "OT"),
    "base2": (GATED, {}, ""),               # the baseline again: what chance alone does
}
TABLE_ORDER = ["base", "S", "O", "T", "SO", "ST", "OT", "SOT", "base2"]     # the order tables show them in
UPGRADED = [a for a in TABLE_ORDER if ARMS[a][2]]                          # the seven arms that are ranked

TRIAL = os.getenv("TRIAL", "0") == "1"
if TRIAL:
    OUT_ROOT, TRIAL_GAMES, TRIAL_ARMS, CAP = "results/screen3_trial", ["ft09", "tu93"], ["base", "SOT", "base2"], 300
else:
    OUT_ROOT = "results/screen3"
SMOKE_ROOT = "results/screen3_smoke"
SPEEDS = os.path.join(SMOKE_ROOT, "speeds.json")
# what a sweep's result depends on: a resume refuses to run if any of it changed since the start
CODE = ["custom_agents", "run_local.py", "compute_metrics.py", "eval_common.py", "metrics_common.py",
        "utils.py", "experiments/goose/screen3.py"]
TRANSITIONS = re.compile(r"^\[run_local\] transitions: (.+)$", re.M)

Job = namedtuple("Job", "game arm cap label")       # label: the manifest's arm column


def game_order():
    """All 25 games with the four tiers taken in turn (dev, seen, A/B-offline,
    untouched, dev, ...), so the first games of a sweep already cover every tier."""
    lists = [list(tiers.TIERS[t]) for t in ("dev", "seen", "ab_offline", "untouched")]
    order = [games[i] for i in range(max(map(len, lists))) for games in lists if i < len(games)]
    if sorted(order) != sorted(benchmark.ALL_GAMES):
        sys.exit("tiers.py and benchmark.ALL_GAMES name different games")
    return order


def sweep_jobs():
    games = TRIAL_GAMES if TRIAL else game_order()
    arms = TRIAL_ARMS if TRIAL else list(ARMS)
    return [Job(g, a, CAP, a) for g in games for a in arms]


def arm_env(arm, cap):
    """The EVAL_* flags of one run of `arm` (EVAL_RESULTS_DIR is added per run)."""
    upgrades, train, _ = ARMS[arm]
    return dict(COMMON, EVAL_UPGRADES=upgrades, EVAL_MAX_ACTIONS=str(cap), **train)


def clean_env():
    """The caller's environment without its EVAL_* settings: each arm sets its own."""
    return {k: v for k, v in os.environ.items() if not k.startswith("EVAL_")}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rel(path):
    return os.path.relpath(path, ROOT)


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ---- one sweep's files -------------------------------------------------------------------------------
def current(must_exist=True):
    """The folder of the sweep the commands act on (results/screen3/CURRENT)."""
    pointer = os.path.join(ROOT, OUT_ROOT, "CURRENT")
    if os.path.exists(pointer):
        return os.path.join(ROOT, open(pointer).read().strip())
    if must_exist:
        sys.exit(f"no sweep yet under {OUT_ROOT}/: use start")
    return None


def done_runs(out):
    """{(game, label): run_dir} for the runs in the manifest (the last row wins)."""
    path = os.path.join(out, "manifest.tsv")
    if not os.path.exists(path):
        return {}
    return {(r["game"], r["arm"]): r["run_dir"] for r in _manifest.read_manifest(path)}


def read_tsv(out, name):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        return []
    return [line.rstrip("\n").split("\t") for line in open(path) if line.strip()]


def worker_pid(out):
    """The sweep's worker, if it is alive (and is this script, not a reused PID)."""
    path = os.path.join(out, "worker.pid")
    if not os.path.exists(path):
        return None
    pid = open(path).read().strip()
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except OSError:
        return None
    return int(pid) if "screen3.py" in cmd and "_work" in cmd else None


# ---- running ---------------------------------------------------------------------------------------
def run_job(out, job, lock):
    """Play one run and score it. Returns None when it is in the manifest, else why not."""
    results_dir = os.path.join(rel(out), "runs", job.label, f"{job.game}_s{SEED}")
    env = clean_env()
    env.update(arm_env(job.arm, job.cap), EVAL_RESULTS_DIR=results_dir)
    logfile = os.path.join(out, "logs", f"{job.label}_{job.game}.log")
    cmd = [sys.executable, "run_local.py", "--game", job.game, "--agent", "goose"] + (["--offline"] if OFFLINE else [])
    try:
        with open(logfile, "w") as f:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"timed out after {RUN_TIMEOUT} s"
    m = TRANSITIONS.search(open(logfile, errors="replace").read())
    if not m:
        return "no recording (the run crashed: see its log)"
    corpus = m.group(1).strip()
    run_dir = os.path.dirname(corpus)
    subprocess.run([sys.executable, "compute_metrics.py", corpus, "--game", job.game,
                    "--agent", f"screen3_{job.label}", "--seed", str(SEED)],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if not os.path.exists(os.path.join(ROOT, run_dir, "metrics.json")):
        return "scoring failed (no metrics.json)"
    try:
        end = json.load(open(os.path.join(ROOT, run_dir, "run_end.json"))).get("termination")
    except (OSError, ValueError):
        end = None
    if end not in ("cap", "win"):
        return f"the run ended with {end!r}, not at its cap or a win"
    with lock:
        if (job.game, job.label) not in done_runs(out):
            with open(os.path.join(out, "manifest.tsv"), "a") as f:
                f.write(f"{run_dir}\t{job.game}\t{SEED}\t{job.label}\n")
    return None


def run_jobs(out, jobs, n_parallel):
    """Run `jobs`, `n_parallel` at a time. Returns (the jobs that failed, paused?).
    A STOP file in the sweep's folder pauses: no new run starts, the runs in
    progress finish."""
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    lock, failed, running = threading.Lock(), [], {}
    stop_file, state_file = os.path.join(out, "STOP"), os.path.join(out, "running.json")

    def save_state():
        with open(state_file, "w") as f:
            json.dump([{"game": j.game, "arm": j.label, "since": t} for j, t in running.items()], f)

    def one(job):
        t0 = time.time()
        try:
            why = run_job(out, job, lock)
        except Exception as e:          # a failed run is never fatal to the sweep
            why = f"{type(e).__name__}: {e}"
        secs = time.time() - t0
        with lock:
            running.pop(job, None)
            save_state()
            with open(os.path.join(out, "timing.tsv"), "a") as f:
                f.write(f"{job.game}\t{job.label}\t{secs:.0f}\t{'ok' if why is None else 'failed'}\n")
            if why is not None:
                failed.append(job)
                with open(os.path.join(out, "failed.tsv"), "a") as f:
                    f.write(f"{job.game}\t{job.label}\t{datetime.now():%Y-%m-%d %H:%M:%S}\t{why}\n")
            n = len(done_runs(out))
        log(f"done  {job.label:<6} {job.game}  {secs:4.0f}s  ({n} in manifest)" if why is None
            else f"!!    {job.label:<6} {job.game}  FAILED: {why}")

    threads, paused = [], False
    for job in jobs:
        while sum(t.is_alive() for t in threads) >= n_parallel:
            time.sleep(2)
        if os.path.exists(stop_file):
            paused = True
            break
        with lock:
            running[job] = datetime.now().strftime("%H:%M:%S")
            save_state()
        t = threading.Thread(target=one, args=(job,))
        t.start()
        threads.append(t)
        log(f"start {job.label:<6} {job.game}")
        time.sleep(1.5)                 # stagger GPU start-up
    for t in threads:
        t.join()
    return failed, paused or os.path.exists(stop_file)


def work(out):
    """The detached worker: the queue, one retry of what failed, then the report."""
    with open(os.path.join(out, "worker.pid"), "w") as f:
        f.write(str(os.getpid()))
    stop_file = os.path.join(out, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    meta = json.load(open(os.path.join(out, "sweep.json")))
    todo = [j for j in sweep_jobs() if (j.game, j.label) not in done_runs(out)]
    log(f"=== screen3 worker: {len(todo)} runs to do, {meta['jobs']} at a time, commit {git('rev-parse', 'HEAD')[:7]}")
    failed, paused = run_jobs(out, todo, meta["jobs"])
    if failed and not paused:
        log(f"=== {len(failed)} runs failed: trying each once more")
        failed, paused = run_jobs(out, failed, meta["jobs"])
    if os.path.exists(stop_file):
        os.remove(stop_file)
    if paused:
        log(f"=== PAUSED with {len(done_runs(out))} runs in the manifest. Continue with: "
            f"uv run python experiments/goose/screen3.py resume")
        return
    left = [j for j in sweep_jobs() if (j.game, j.label) not in done_runs(out)]
    log(f"=== queue empty: {len(done_runs(out))} runs in the manifest, {len(left)} missing. Writing the report.")
    try:
        import screen3_report
        screen3_report.write(out)
    except Exception as e:              # the runs are in the manifest whatever happens to the tables
        log(f"!!    the report failed ({type(e).__name__}: {e}); run it by hand: screen3.py report")
    log("=== DONE" if not left else f"=== DONE, WITH {len(left)} RUNS MISSING (see failed.tsv; resume tries them again)")


def launch(out):
    """Start the worker detached from this terminal, with no EVAL_* setting of the caller's."""
    logf = open(os.path.join(out, "worker.log"), "a")
    subprocess.Popen(["setsid", "nohup", sys.executable, os.path.abspath(__file__), "_work", out],
                     cwd=ROOT, env=clean_env(), stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    time.sleep(1.5)
    print(f"started detached (pid {worker_pid(out)}); log: {rel(out)}/worker.log")
    print("  status: uv run python experiments/goose/screen3.py status      pause: ... pause      continue: ... resume")


# ---- the commands ----------------------------------------------------------------------------------
def estimate(jobs, n_parallel):
    """Hours the jobs should take, from the speeds the smoke run measured (or None)."""
    path = os.path.join(ROOT, SPEEDS)
    if not os.path.exists(path):
        return None, None
    sp = json.load(open(path))
    secs = sum(sp["overhead_s"] + j.cap * sp["ms_per_action"].get(j.arm, sp["ms_per_action"]["base"]) / 1000
               for j in jobs)
    return secs / n_parallel / 3600, sp


def cmd_plan(a):
    jobs = sweep_jobs()
    print(f"UPGRADE SCREEN 3{' (TRIAL)' if TRIAL else ''}: {len({j.arm for j in jobs})} arms x "
          f"{len({j.game for j in jobs})} games x seed {SEED} x {CAP:,} actions = {len(jobs)} runs, {a.jobs} at a time")
    print(f"every run: run_local.py --agent goose{' --offline' if OFFLINE else ''}, with "
          + " ".join(f"{k}={v}" for k, v in COMMON.items()) + f" EVAL_MAX_ACTIONS={CAP}")
    print(f"\n  {'arm':<6} {'EVAL_UPGRADES':<34} training flags")
    for arm in TABLE_ORDER:
        upgrades, train, _ = ARMS[arm]
        print(f"  {arm:<6} {upgrades:<34} {' '.join(f'{k}={v}' for k, v in train.items()) or 'none'}")
    print(f"\nrun order: games outside ({' '.join(dict.fromkeys(j.game for j in jobs))}),")
    print(f"           arms inside ({' '.join(dict.fromkeys(j.arm for j in jobs))})")
    hours, sp = estimate(jobs, a.jobs)
    if hours is None:
        print(f"\ntime: no estimate yet (run `smoke` first: it writes {SPEEDS})")
    else:
        ms = sp["ms_per_action"]
        print(f"\ntime: about {hours:.1f} h, from the smoke run of {sp['measured']} "
              f"({sp['jobs']} runs at a time): ms per action "
              + ", ".join(f"{arm} {ms[arm]:.0f}" for arm in TABLE_ORDER if arm in ms)
              + f"; {sp['overhead_s']:.0f} s per run for start-up and scoring")


def cmd_start(a):
    cur = current(must_exist=False)
    if cur and worker_pid(cur):
        sys.exit(f"a sweep is already running (pid {worker_pid(cur)}): see status")
    if cur and len(done_runs(cur)) < len(sweep_jobs()) and not a.new:
        sys.exit(f"a sweep was already started ({rel(cur)}, {len(done_runs(cur))} of {len(sweep_jobs())} runs): "
                 f"use resume, or start --new to begin another")
    dirty = git("status", "--porcelain", "--ignore-submodules=all")
    if dirty and not TRIAL:
        sys.exit("the sweep starts only from committed code. Uncommitted or untracked:\n" + dirty)
    out = os.path.join(ROOT, OUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out, "logs"))
    with open(os.path.join(out, "sweep.json"), "w") as f:
        json.dump({"started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "commit": git("rev-parse", "HEAD"),
                   "dirty": bool(dirty), "jobs": a.jobs, "seed": SEED, "cap": CAP, "offline": OFFLINE,
                   "common": COMMON, "arms": {arm: arm_env(arm, CAP) for arm in ARMS},
                   "games": list(dict.fromkeys(j.game for j in sweep_jobs()))}, f, indent=1)
    with open(os.path.join(ROOT, OUT_ROOT, "CURRENT"), "w") as f:
        f.write(rel(out) + "\n")
    print(f"sweep folder: {rel(out)}   commit {git('rev-parse', 'HEAD')[:7]}")
    launch(out)


def cmd_resume(a):
    out = current()
    if worker_pid(out):
        sys.exit(f"the sweep is running (pid {worker_pid(out)}): nothing to resume")
    left = [j for j in sweep_jobs() if (j.game, j.label) not in done_runs(out)]
    if not left:
        sys.exit("every run is in the manifest: use report")
    meta = json.load(open(os.path.join(out, "sweep.json")))
    if meta["arms"] != {arm: arm_env(arm, CAP) for arm in ARMS}:
        sys.exit("the arms in this script are not the arms the sweep started with (sweep.json): "
                 "an arm never changes once a sweep has started")
    changed = git("diff", "--name-only", meta["commit"], "--", *CODE)
    if changed and not TRIAL:
        sys.exit(f"code the runs depend on changed since the sweep started at {meta['commit'][:7]}:\n{changed}\n"
                 "the rest of the sweep would not be the same agent. Check out that code, or start --new.")
    with open(os.path.join(out, "resumes.txt"), "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{git('rev-parse', 'HEAD')}\t{len(left)} runs left\n")
    print(f"resuming {rel(out)}: {len(left)} of {len(sweep_jobs())} runs left")
    launch(out)


def cmd_pause(a):
    out = current()
    if not worker_pid(out):
        sys.exit("the sweep is not running: nothing to pause")
    open(os.path.join(out, "STOP"), "w").close()
    print("pausing: no new run starts, and the sweep stops when the runs in progress finish (a few minutes).")
    print("  see it with: status      continue with: resume")


def cmd_status(a):
    out = current()
    jobs, done = sweep_jobs(), done_runs(out)
    pid = worker_pid(out)
    print(f"screen3: {len(done)} of {len(jobs)} runs done; the sweep is "
          f"{'RUNNING (pid %d)' % pid if pid else 'NOT running'}   [{rel(out)}]")
    print("  per arm: " + "  ".join(f"{arm} {sum(k[1] == arm for k in done)}" for arm in dict.fromkeys(j.arm for j in jobs)))
    if pid and os.path.exists(os.path.join(out, "STOP")):
        print("  pausing: it stops when the runs in progress finish")
    state = os.path.join(out, "running.json")
    if pid and os.path.exists(state):
        try:
            print("  in progress: " + ", ".join(f"{r['arm']} {r['game']} (since {r['since']})" for r in json.load(open(state))))
        except ValueError:
            pass
    left = [j for j in jobs if (j.game, j.label) not in done]
    fails = [r for r in read_tsv(out, "failed.tsv") if (r[0], r[1]) not in done]
    if fails:
        print(f"  failed and not yet in the manifest: {len(fails)} (" + ", ".join(sorted({f'{r[1]} {r[0]}' for r in fails})[:8]) + ")")
    took = {}
    for game, label, secs, ok in read_tsv(out, "timing.tsv"):
        if ok == "ok":
            took.setdefault(label, []).append(float(secs))
    if left and took:
        every = st.mean(s for v in took.values() for s in v)
        meta = json.load(open(os.path.join(out, "sweep.json")))
        hours = sum(st.mean(took[j.label]) if j.label in took else every for j in left) / meta["jobs"] / 3600
        print(f"  about {hours:.1f} h of running left (a finished run took {every / 60:.1f} min on average, "
              f"{meta['jobs']} at a time)")
    if not left:
        print(f"  every run is in: uv run python experiments/goose/screen3.py report   (tables: {rel(out)}/report.md)")


def cmd_report(a):
    import screen3_report
    out = os.path.join(ROOT, a.sweep) if a.sweep else current()
    print(screen3_report.write(out, docs=a.docs))


# ---- the smoke runs (check 4 of the plan) ------------------------------------------------------------
def mean_ms(run_dir, skip=100):
    """Mean wall-clock ms per recorded move, after the first `skip` (model start-up)."""
    import glob
    import numpy as np
    ms = np.concatenate([np.load(p)["wall_ms"] for p in
                         sorted(glob.glob(os.path.join(ROOT, run_dir, "transitions", "shard_*.npz")))])
    return float(ms[skip:].mean()) if len(ms) > skip else float("nan")


def load_json(run_dir, name):
    path = os.path.join(ROOT, run_dir, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def cmd_smoke(a):
    """(a) arm SOT on all 25 games, 300 actions: no crash, every game loads. (b) all
    eight settings on ft09 and tu93, 1,000 actions: each upgrade acted; speed per arm.
    --evaluate FOLDER prints the checks again for a smoke run already made."""
    part_a = [Job(g, "SOT", 300, "a_SOT") for g in game_order()]
    part_b = [Job(g, arm, 1000, "b_" + arm) for g in ("ft09", "tu93") for arm in list(ARMS)[:-1]]
    if a.evaluate:
        out = os.path.join(ROOT, a.evaluate)
        todo = [j for j in part_a + part_b if any(label == j.label for _, label in done_runs(out))]
        print(f"smoke: the checks again, on the {len(done_runs(out))} runs under {rel(out)}")
    else:
        out = os.path.join(ROOT, SMOKE_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(out, "logs"))
        todo = (part_a if a.part in ("a", "ab") else []) + (part_b if a.part in ("b", "ab") else [])
        print(f"smoke: {len(todo)} runs, {a.jobs} at a time, under {rel(out)}", flush=True)
        t0 = time.time()
        run_jobs(out, todo, a.jobs)
        print(f"\nsmoke finished in {(time.time() - t0) / 60:.1f} min")
    done = done_runs(out)
    print(f"{len(done)} of {len(todo)} runs played and scored")
    ok = True
    if part_a[0] in todo:
        bad = [j.game for j in part_a if (j.game, j.label) not in done]
        ok &= not bad
        print(f"(a) SOT on 25 games x 300 actions{' --offline' if OFFLINE else ''}: "
              + ("PASS, no crash on any game" if not bad else f"FAIL on {bad}"))
        slow = [mean_ms(done[(j.game, j.label)]) for j in part_a if (j.game, j.label) in done]
        if slow:                        # every run beside it was a T arm too: the slowest the T arms get
            print(f"    SOT with {a.jobs} SOT runs at a time: {st.mean(slow):.1f} ms per action "
                  f"(slowest game {max(slow):.1f}), after each run's first 100 moves")
    if part_b[0] in todo:
        ms, over, rows = {}, [], []
        timing = {(g, label): float(secs) for g, label, secs, _ in read_tsv(out, "timing.tsv")}
        for arm in list(ARMS)[:-1]:
            per = {}
            for g in ("ft09", "tu93"):
                d = done.get((g, "b_" + arm))
                if d is None:
                    continue
                m, up, mp, tr = (load_json(d, n) for n in ("metrics.json", "upgrades_stats.json",
                                                              "return_map_stats.json", "train_stats.json"))
                per[g] = dict(ms=mean_ms(d), med=m.get("wall_ms_med"), aps=load_json(d, "run_end.json").get("actions_per_sec"),
                              redrawn=up.get("objclick_redrawn"), steps=tr.get("train_steps"),
                              scheds=sum(1 for k, v in mp.items() if k.startswith("sched_pulls_") and v > 0),
                              untried=mp.get("untried_first"))
                over.append(timing[(g, "b_" + arm)] - 1000 * per[g]["ms"] / 1000)
            if len(per) == 2:
                ms[arm] = st.mean(p["ms"] for p in per.values())
                rows.append((arm, per))
        print("(b) eight settings on ft09 and tu93 x 1,000 actions (ft09 | tu93):")
        print(f"    {'arm':<5} {'mean ms/action':>14}  {'act/s':>13}  {'wall_ms_med':>13}  {'clicks redrawn':>14}  "
              f"{'schedules used':>14}  {'untried-first':>13}  {'training steps':>14}")
        show = lambda f, t, k: f"{'-' if f[k] is None else f[k]} | {'-' if t[k] is None else t[k]}"
        for arm, per in rows:
            f, t = per["ft09"], per["tu93"]
            print(f"    {arm:<5} {ms[arm]:>14.1f}  {show(f, t, 'aps'):>13}  {show(f, t, 'med'):>13}  {show(f, t, 'redrawn'):>14}  "
                  f"{show(f, t, 'scheds'):>14}  {show(f, t, 'untried'):>13}  {show(f, t, 'steps'):>14}")
        has = lambda arm, u: u in ARMS[arm][2]
        med = lambda with_t: [per[g]["med"] for arm, per in rows if has(arm, "T") == with_t for g in per]
        ratio = st.median(med(True)) / st.median(med(False)) if rows else 0.0
        checks = {
            "all 16 runs finished": len(rows) == 8,
            "O redraws clicks on ft09 and none on tu93 (no clicks there)":
                all(per["ft09"]["redrawn"] and per["tu93"]["redrawn"] == 0 for arm, per in rows if has(arm, "O")),
            "the other arms redraw nothing":
                all(per[g]["redrawn"] is None for arm, per in rows if not has(arm, "O") for g in per),
            "S used at least two schedules on both games":
                all(per[g]["scheds"] >= 2 for arm, per in rows if has(arm, "S") for g in per),
            "the other arms used no schedule":
                all(per[g]["scheds"] == 0 for arm, per in rows if not has(arm, "S") for g in per),
            "T took a training step on at least 7 actions in 10 (an arm without it: 2 in 10 at most)":
                all((per[g]["steps"] or 0) >= 700 for arm, per in rows if has(arm, "T") for g in per),
            "the other arms hold nothing from train_more":
                all(per[g]["steps"] is None for arm, per in rows if not has(arm, "T") for g in per),
            f"T's wall_ms_med is several times the others' (median of the T runs / median of the rest = {ratio:.1f})":
                ratio >= 2.5,
        }
        for name, good in checks.items():
            ok &= bool(good)
            print(f"    {'PASS' if good else 'FAIL'}: {name}")
        if len(rows) == 8:
            with open(os.path.join(ROOT, SPEEDS), "w") as f:
                json.dump({"measured": datetime.now().strftime("%Y-%m-%d %H:%M"), "jobs": a.jobs, "smoke": rel(out),
                           "ms_per_action": dict(ms, base2=ms["base"]), "overhead_s": max(st.mean(over), 0.0)}, f, indent=1)
            print(f"    speeds written to {SPEEDS}; `plan` now gives a time estimate")
    for g, label, when, why in read_tsv(out, "failed.tsv"):
        print(f"    failed: {label} {g}: {why}")
    print("SMOKE: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, text in (("plan", cmd_plan, "the arms, their flags, the run count, a time estimate"),
                           ("smoke", cmd_smoke, "the GPU checks before the sweep"),
                           ("start", cmd_start, "begin the sweep, detached"),
                           ("status", cmd_status, "runs done, runs in progress, time left"),
                           ("pause", cmd_pause, "stop when the runs in progress finish"),
                           ("resume", cmd_resume, "continue after a pause, a crash or a reboot"),
                           ("report", cmd_report, "write the tables (also on a partial manifest)")):
        p = sub.add_parser(name, help=text)
        p.set_defaults(fn=fn)
        if name in ("plan", "smoke", "start"):
            p.add_argument("--jobs", type=int, default=JOBS, help="runs at a time (default 4)")
        if name == "smoke":
            p.add_argument("--part", choices=("a", "b", "ab"), default="ab")
            p.add_argument("--evaluate", default="", metavar="FOLDER",
                           help="print the checks again for a smoke run already made (no run is started)")
        if name == "start":
            p.add_argument("--new", action="store_true", help="begin another sweep although one is unfinished")
        if name == "report":
            p.add_argument("--sweep", default="", help="a sweep folder (default: the current one)")
            p.add_argument("--docs", action="store_true",
                           help="also write docs/1-goose-upgrades/upgrade-screen3-runs.csv (the backup of the numbers)")
    w = sub.add_parser("_work")         # internal: the detached worker
    w.add_argument("out")
    w.set_defaults(fn=lambda a: work(a.out))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
