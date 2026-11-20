#!/usr/bin/env python3
"""kp_test.py - the Knowledge Planner's GPU runs: the smoke, the dev sweep and the test.

The problem it fixes: the test is 150 runs and about 5.5 hours of GPU time with
nobody watching, and its arms must be exactly the ones its page names
(docs/4-knowledge-planner/kp-test.md). This starts the runs detached, 4 at a
time, says how far they are, pauses and resumes them, and retries a failed run
once. Plan: docs/4-knowledge-planner/kp-plan.md, sections 5 and 6.

    uv run python experiments/kp/kp_test.py plan      the arms, their flags, the run count
    uv run python experiments/kp/kp_test.py smoke     6 runs, about 15 minutes of GPU: ft09, lp85 and tu93
                                                      x seed 14 x SOT and SOT-KP, then the checks of Phase 1.5
    uv run python experiments/kp/kp_test.py start     begin, detached (it outlives the terminal)
    uv run python experiments/kp/kp_test.py status    runs done, runs in progress, time left
    uv run python experiments/kp/kp_test.py pause     no new run starts; the runs in progress finish, then it stops
    uv run python experiments/kp/kp_test.py resume    continue after a pause, a crash or a reboot
    uv run python experiments/kp/kp_test.py verdict   the rule and the tables (experiments/kp/kp_verdict.py), once
                                                      every run is in; the dev sweep has tables and no rule

Without DEV=1 the commands act on THE TEST: three arms x all 25 games x seeds
11 and 12 x 5,000 actions = 150 runs, under results/kp/test/. With DEV=1 they
act on the dev sweep: SOT and SOT-KP x the carry test's seven games x seeds 14
and 15 = 28 runs, under results/kp/dev/.

The arms, all on Goose with upgrade screen 3's three upgrades (`SOT`), set by
flags as screen 3 set them (no preset exists, and a preset refuses other flags):

    SOT      Goose with the three upgrades
    SOT-KP   the same with the knowledge planner (EVAL_KP=1, experiments/kp/kp.toml)
    SOT2     SOT again: how far two runs of the same agent come apart

Every run is `run_local.py --agent goose --offline` with its arm's flags and
none of the caller's EVAL_* settings. Each run has its own EVAL_RESULTS_DIR
under <sweep>/runs/<arm>/<game>_s<seed>/, and enters manifest.tsv (run_dir
<TAB> game <TAB> seed <TAB> arm) only after it has been played to its cap, or
to the end of the game, and scored. So a pause loses nothing, and a crash or a
reboot loses only the runs in progress. A failed run goes to failed.tsv and is
tried once more when the queue is empty.

The test starts only from committed code. A resume refuses to go on if the arms
or the code the runs depend on (CODE) changed since the start: the rest of the
sweep would not be the same agent. Games are the outer loop with the four tiers
of tiers.py taken in turn, and the arms run back to back inside a game and
seed, so a sweep cut short is still paired and still covers every tier.

TRIAL=<actions> does all of this on tiny runs under results/kp/trial/, to try
the launcher itself (start, pause, resume, the smoke's checks) before a real
sweep; with CUDA_VISIBLE_DEVICES= in front it needs no GPU.

Modelled on experiments/goose/screen3.py; it imports nothing from it.
"""
import argparse
import hashlib
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
for _p in (ROOT, os.path.join(ROOT, "custom_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark  # noqa: E402
import manifest as _manifest  # noqa: E402  (the repo's one manifest reader)
import tiers  # noqa: E402

# ---- the run set -----------------------------------------------------------------------------------
CAP = 5000
JOBS = 4                    # runs at a time (the GPU is shared: 4 is the measured best)
OFFLINE = True
RUN_TIMEOUT = 7200          # seconds; a run that takes longer is killed and counted as failed. A run takes
                            # about 9 minutes; the margin is for a planner run on a game nobody has seen,
                            # because a killed run leaves the test without a verdict
SOT = {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1", "EVAL_UPGRADES": "bars,map_sched,attempt,objclick",
       "EVAL_TRAIN_EVERY": "1", "EVAL_TRAIN_RECENT": "16", "PYTHONHASHSEED": "0"}
KP_CONFIG = "experiments/kp/kp.toml"
ARMS = {"SOT": {}, "SOT-KP": {"EVAL_KP": "1", "EVAL_KP_CONFIG": KP_CONFIG}, "SOT2": {}}   # in the order they run

DEV = os.getenv("DEV", "0") == "1"
DEV_GAMES = ("vc33", "ft09", "cd82", "dc22", "m0r0", "lp85", "tu93")    # the carry test's seven; tu93 has no clicks
if DEV:
    NAME, OUT_ROOT, SEEDS, ARM_NAMES = "the dev sweep", "results/kp/dev", (14, 15), ("SOT", "SOT-KP")
else:
    NAME, OUT_ROOT, SEEDS, ARM_NAMES = "THE TEST", "results/kp/test", (11, 12), ("SOT", "SOT-KP", "SOT2")
SMOKE_ROOT, SMOKE_GAMES, SMOKE_SEED, SMOKE_ARMS = "results/kp/smoke", ("ft09", "lp85", "tu93"), 14, ("SOT", "SOT-KP")
TRIAL = int(os.getenv("TRIAL", "0"))
if TRIAL:
    NAME, CAP = f"{NAME} (TRIAL: {TRIAL} actions a run)", TRIAL
    OUT_ROOT, SMOKE_ROOT = OUT_ROOT.replace("results/kp/", "results/kp/trial/"), "results/kp/trial/smoke"
# what a run's result depends on: a resume refuses to go on if any of it changed since the start
CODE = ["custom_agents", "run_local.py", "compute_metrics.py", "eval_common.py", "metrics_common.py", "utils.py",
        "experiments/kp/kp_test.py", KP_CONFIG]
TRANSITIONS = re.compile(r"^\[run_local\] transitions: (.+)$", re.M)

Job = namedtuple("Job", "game seed arm cap")


def game_order():
    """All 25 games with the four tiers taken in turn (dev, seen, A/B-offline,
    untouched, dev, ...), so the first games of a sweep already cover every tier."""
    lists = [list(tiers.TIERS[t]) for t in ("dev", "seen", "ab_offline", "untouched")]
    order = [games[i] for i in range(max(map(len, lists))) for games in lists if i < len(games)]
    if sorted(order) != sorted(benchmark.ALL_GAMES):
        sys.exit("tiers.py and benchmark.ALL_GAMES name different games")
    return order


def sweep_jobs():
    games = [g for g in game_order() if g in DEV_GAMES] if DEV else game_order()
    return [Job(g, s, a, CAP) for g in games for s in SEEDS for a in ARM_NAMES]


def arm_env(arm, seed, cap=CAP):
    """The flags of one run of `arm` (EVAL_RESULTS_DIR is added per run)."""
    return dict(SOT, EVAL_SEED=str(seed), EVAL_MAX_ACTIONS=str(cap), **ARMS[arm])


def clean_env():
    """The caller's environment without its EVAL_* settings: each arm sets its own."""
    return {k: v for k, v in os.environ.items() if not k.startswith("EVAL_")}


def code_hash():
    """One hash over every .py and .toml file the runs depend on (CODE)."""
    paths = []
    for entry in CODE:
        full = os.path.join(ROOT, entry)
        if os.path.isdir(full):
            for d, dirs, files in os.walk(full):
                dirs[:] = sorted(x for x in dirs if x != "__pycache__")
                paths += [os.path.join(d, f) for f in files if f.endswith((".py", ".toml"))]
        else:
            paths.append(full)
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(os.path.relpath(path, ROOT).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def kp_config_hash():
    """The hash a planner run writes into run_config.json for the committed settings."""
    import kp_bridge
    return kp_bridge.digest(kp_bridge.load_settings(os.path.join(ROOT, KP_CONFIG)))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rel(path):
    return os.path.relpath(path, ROOT)


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ---- one sweep's files -------------------------------------------------------------------------------
def current(must_exist=True):
    """The folder of the sweep the commands act on (<OUT_ROOT>/CURRENT)."""
    pointer = os.path.join(ROOT, OUT_ROOT, "CURRENT")
    if os.path.exists(pointer):
        return os.path.join(ROOT, open(pointer).read().strip())
    if must_exist:
        sys.exit(f"no sweep yet under {OUT_ROOT}/: use start")
    return None


def done_runs(out):
    """{(game, seed, arm): run_dir} for the runs in the manifest (the last row wins)."""
    path = os.path.join(out, "manifest.tsv")
    if not os.path.exists(path):
        return {}
    return {(r["game"], int(r["seed"]), r["arm"]): r["run_dir"] for r in _manifest.read_manifest(path)}


def key(job):
    return (job.game, job.seed, job.arm)


def read_tsv(out, name):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        return []
    return [line.rstrip("\n").split("\t") for line in open(path) if line.strip()]


def load_json(run_dir, name):
    path = os.path.join(ROOT, run_dir, name)
    return json.load(open(path)) if os.path.exists(path) else {}


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
    return int(pid) if "kp_test.py" in cmd and "_work" in cmd else None


# ---- running ---------------------------------------------------------------------------------------
def run_job(out, job, lock):
    """Play one run and score it. Returns None when it is in the manifest, else why not."""
    tag = f"{job.arm}_{job.game}_s{job.seed}"
    env = clean_env()
    env.update(arm_env(job.arm, job.seed, job.cap),
               EVAL_RESULTS_DIR=os.path.join(rel(out), "runs", job.arm, f"{job.game}_s{job.seed}"))
    logfile = os.path.join(out, "logs", f"{tag}.log")
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
    run_dir = os.path.dirname(corpus)       # as run_local.py printed it: under EVAL_RESULTS_DIR, so relative to the repo
    run_dir = rel(run_dir) if os.path.isabs(run_dir) else run_dir
    subprocess.run([sys.executable, "compute_metrics.py", corpus, "--game", job.game,
                    "--agent", f"kp_{job.arm}", "--seed", str(job.seed)],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if not os.path.exists(os.path.join(ROOT, run_dir, "metrics.json")):
        return "scoring failed (no metrics.json)"
    end = load_json(run_dir, "run_end.json").get("termination")
    if end not in ("cap", "win"):
        return f"the run ended with {end!r}, not at its cap or a win"
    with lock:
        if key(job) not in done_runs(out):
            with open(os.path.join(out, "manifest.tsv"), "a") as f:
                f.write(f"{run_dir}\t{job.game}\t{job.seed}\t{job.arm}\n")
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
            json.dump([{"game": j.game, "seed": j.seed, "arm": j.arm, "since": t} for j, t in running.items()], f)

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
                f.write(f"{job.game}\t{job.seed}\t{job.arm}\t{secs:.0f}\t{'ok' if why is None else 'failed'}\n")
            if why is not None:
                failed.append(job)
                with open(os.path.join(out, "failed.tsv"), "a") as f:
                    f.write(f"{job.game}\t{job.seed}\t{job.arm}\t{datetime.now():%Y-%m-%d %H:%M:%S}\t{why}\n")
            n = len(done_runs(out))
        log(f"done  {job.arm:<7} {job.game} s{job.seed}  {secs:4.0f}s  ({n} in manifest)" if why is None
            else f"!!    {job.arm:<7} {job.game} s{job.seed}  FAILED: {why}")

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
        log(f"start {job.arm:<7} {job.game} s{job.seed}")
        time.sleep(1.5)                 # stagger GPU start-up
    for t in threads:
        t.join()
    return failed, paused or os.path.exists(stop_file)


def work(out):
    """The detached worker: the queue, then one retry of what failed."""
    with open(os.path.join(out, "worker.pid"), "w") as f:
        f.write(str(os.getpid()))
    stop_file = os.path.join(out, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    meta = json.load(open(os.path.join(out, "sweep.json")))
    todo = [j for j in sweep_jobs() if key(j) not in done_runs(out)]
    log(f"=== kp_test worker ({NAME}): {len(todo)} runs to do, {meta['jobs']} at a time, commit {git('rev-parse', 'HEAD')[:7]}")
    failed, paused = run_jobs(out, todo, meta["jobs"])
    if failed and not paused:
        log(f"=== {len(failed)} runs failed: trying each once more")
        failed, paused = run_jobs(out, failed, meta["jobs"])
    if os.path.exists(stop_file):
        os.remove(stop_file)
    if paused:
        log(f"=== PAUSED with {len(done_runs(out))} runs in the manifest. Continue with: resume")
        return
    left = [j for j in sweep_jobs() if key(j) not in done_runs(out)]
    log(f"=== DONE: every run is in the manifest ({len(done_runs(out))})" if not left
        else f"=== DONE, WITH {len(left)} RUNS MISSING (see failed.tsv; resume tries them again)")


def launch(out):
    """Start the worker detached from this terminal, with no EVAL_* setting of the caller's."""
    logf = open(os.path.join(out, "worker.log"), "a")
    subprocess.Popen(["setsid", "nohup", sys.executable, os.path.abspath(__file__), "_work", out],
                     cwd=ROOT, env=clean_env(), stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    time.sleep(1.5)
    print(f"started detached (pid {worker_pid(out)}); log: {rel(out)}/worker.log")
    print("  status, pause and resume: the same command with that word" + ("  (keep DEV=1)" if DEV else ""))


# ---- the commands ----------------------------------------------------------------------------------
def cmd_plan(a):
    jobs = sweep_jobs()
    games = list(dict.fromkeys(j.game for j in jobs))
    print(f"{NAME}: {len(ARM_NAMES)} arms x {len(games)} games x seeds {', '.join(map(str, SEEDS))} x {CAP:,} actions "
          f"= {len(jobs)} runs, {a.jobs} at a time, under {OUT_ROOT}/")
    print(f"every run: run_local.py --agent goose{' --offline' if OFFLINE else ''}, with EVAL_SEED=<seed> "
          f"EVAL_MAX_ACTIONS={CAP} " + " ".join(f"{k}={v}" for k, v in SOT.items()))
    for arm in ARM_NAMES:
        print(f"  {arm:<7} " + (" ".join(f"{k}={v}" for k, v in ARMS[arm].items()) or "nothing more"))
    print(f"planner settings: {KP_CONFIG}, hash {kp_config_hash()}   code the runs depend on: hash {code_hash()}")
    print(f"run order: games outside ({' '.join(games)}), then seeds, then arms ({' '.join(ARM_NAMES)})")
    print("time: about 8.5 minutes a run with 4 at a time (upgrade screen 3's SOT runs): "
          f"about {len(jobs) * 8.5 / a.jobs / 60:.1f} h")


def cmd_start(a):
    cur = current(must_exist=False)
    if cur and worker_pid(cur):
        sys.exit(f"a sweep is already running (pid {worker_pid(cur)}): see status")
    if cur and len(done_runs(cur)) < len(sweep_jobs()) and not a.new:
        sys.exit(f"a sweep was already started ({rel(cur)}, {len(done_runs(cur))} of {len(sweep_jobs())} runs): "
                 f"use resume, or start --new to begin another")
    dirty = git("status", "--porcelain", "--ignore-submodules=all")
    if dirty and not (DEV or TRIAL):
        sys.exit("the test starts only from committed code. Uncommitted or untracked:\n" + dirty)
    out = os.path.join(ROOT, OUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out, "logs"))
    with open(os.path.join(out, "sweep.json"), "w") as f:
        json.dump({"what": NAME, "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "commit": git("rev-parse", "HEAD"),
                   "dirty": bool(dirty), "code_hash": code_hash(), "kp_config_hash": kp_config_hash(), "jobs": a.jobs,
                   "seeds": list(SEEDS), "cap": CAP, "offline": OFFLINE,
                   "arms": {arm: arm_env(arm, "<seed>") for arm in ARM_NAMES},
                   "games": list(dict.fromkeys(j.game for j in sweep_jobs()))}, f, indent=1)
    with open(os.path.join(ROOT, OUT_ROOT, "CURRENT"), "w") as f:
        f.write(rel(out) + "\n")
    print(f"{NAME}: {len(sweep_jobs())} runs, folder {rel(out)}, commit {git('rev-parse', 'HEAD')[:7]}"
          f"{' (with uncommitted changes)' if dirty else ''}")
    launch(out)


def cmd_resume(a):
    out = current()
    if worker_pid(out):
        sys.exit(f"the sweep is running (pid {worker_pid(out)}): nothing to resume")
    left = [j for j in sweep_jobs() if key(j) not in done_runs(out)]
    if not left:
        sys.exit("every run is in the manifest")
    meta = json.load(open(os.path.join(out, "sweep.json")))
    if meta["arms"] != {arm: arm_env(arm, "<seed>") for arm in ARM_NAMES}:
        sys.exit("the arms in this script are not the arms the sweep started with (sweep.json): "
                 "an arm never changes once a sweep has started")
    if meta["code_hash"] != code_hash():
        sys.exit(f"code the runs depend on changed since the sweep started (commit {meta['commit'][:7]}): the rest of "
                 f"the sweep would not be the same agent. Check out that code, or start --new.")
    with open(os.path.join(out, "resumes.txt"), "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{git('rev-parse', 'HEAD')}\t{len(left)} runs left\n")
    print(f"resuming {rel(out)}: {len(left)} of {len(sweep_jobs())} runs left")
    launch(out)


def cmd_pause(a):
    out = current()
    if not worker_pid(out):
        sys.exit("the sweep is not running: nothing to pause")
    open(os.path.join(out, "STOP"), "w").close()
    print("pausing: no new run starts, and the sweep stops when the runs in progress finish (up to ten minutes).")


def cmd_status(a):
    out = current()
    jobs, done, pid = sweep_jobs(), done_runs(out), worker_pid(out)
    print(f"{NAME}: {len(done)} of {len(jobs)} runs done; the sweep is "
          f"{'RUNNING (pid %d)' % pid if pid else 'NOT running'}   [{rel(out)}]")
    print("  per arm: " + "  ".join(f"{arm} {sum(k[2] == arm for k in done)}" for arm in ARM_NAMES))
    if pid and os.path.exists(os.path.join(out, "STOP")):
        print("  pausing: it stops when the runs in progress finish")
    state = os.path.join(out, "running.json")
    if pid and os.path.exists(state):
        try:
            print("  in progress: " + ", ".join(f"{r['arm']} {r['game']} s{r['seed']} (since {r['since']})"
                                                for r in json.load(open(state))))
        except ValueError:
            pass
    left = [j for j in jobs if key(j) not in done]
    fails = [r for r in read_tsv(out, "failed.tsv") if (r[0], int(r[1]), r[2]) not in done]
    if fails:
        print(f"  failed and not yet in the manifest: {len(fails)} ("
              + ", ".join(sorted({f'{r[2]} {r[0]} s{r[1]}' for r in fails})[:8]) + ")")
    took = [float(r[3]) for r in read_tsv(out, "timing.tsv") if r[4] == "ok"]
    if left and took:
        meta = json.load(open(os.path.join(out, "sweep.json")))
        print(f"  about {len(left) * st.mean(took) / meta['jobs'] / 3600:.1f} h of running left "
              f"(a finished run took {st.mean(took) / 60:.1f} min on average, {meta['jobs']} at a time)")
    if not left:
        print("  every run is in the manifest")


def cmd_verdict(a):
    """The rule lives in kp_verdict.py, which is not among CODE: a fix to a table never blocks a resume."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kp_verdict
    print(kp_verdict.write(current(), a.partial))


# ---- the smoke (Phase 1.5 of the plan) ---------------------------------------------------------------
def cmd_smoke(a):
    """Six runs, then the checks. --evaluate FOLDER prints the checks again for a smoke already made."""
    jobs = [Job(g, SMOKE_SEED, arm, CAP) for g in SMOKE_GAMES for arm in SMOKE_ARMS]
    if a.evaluate:
        out = os.path.join(ROOT, a.evaluate)
    else:
        out = os.path.join(ROOT, SMOKE_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(out, "logs"))
        with open(os.path.join(out, "sweep.json"), "w") as f:
            json.dump({"what": "the smoke", "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--ignore-submodules=all")),
                       "code_hash": code_hash(), "kp_config_hash": kp_config_hash(), "jobs": a.jobs}, f, indent=1)
        print(f"smoke: {len(jobs)} runs, {a.jobs} at a time, under {rel(out)}", flush=True)
        t0 = time.time()
        run_jobs(out, jobs, a.jobs)
        print(f"\nsmoke finished in {(time.time() - t0) / 60:.1f} min")
    done, rows = done_runs(out), {}
    for j in jobs:
        d = done.get(key(j))
        if d is None:
            continue
        end, cfg, tr, kp, m = (load_json(d, n) for n in ("run_end.json", "run_config.json", "train_stats.json",
                                                         "kp/kp_stats.json", "metrics.json"))
        rows[j.game, j.arm] = dict(levels=end["final_levels"], ended=end["termination"], actions=end["n_actions"],
                                   ms=1000 * end["elapsed_s"] / end["n_actions"], cfg=cfg, kp=kp,
                                   trained=tr.get("train_steps", 0) / end["n_actions"],
                                   ups=[int(x) for x, _ in m.get("levelup_events") or []])
    print(f"\n  {'game':<5} {'arm':<7} {'levels':>6}  {'level-ups at action':<28} {'actions':>7} {'ms/action':>9} "
          f"{'trained':>7}  the planner")
    for (game, arm), r in rows.items():
        kp = r["kp"]
        said = "" if not kp else ("dormant (no click action)" if kp["dormant"] else (
            f"{kp['levels_by_planning']} levels by planning, clicks {kp['clicks']}, "
            f"{kp['fallbacks']} fallbacks, {kp['reference_seconds']:.0f} s in the reference, errors {kp['kp_errors']}"))
        print(f"  {game:<5} {arm:<7} {r['levels']:>6}  {' '.join(map(str, r['ups'])):<28} {r['actions']:>7} "
              f"{r['ms']:>9.1f} {r['trained']:>7.0%}  {said}")
    for game, arm in rows:
        for v in rows[game, arm]["kp"].get("levels", []) if rows[game, arm]["kp"] else []:
            if v["gave_up"]:
                print(f"    {game} level {v['level']}: the planner gave up at action {v['gave_up_at']}: {v['gave_up']}")
    sot = lambda c: (c.get("label") == "novel" and c.get("return_map") and c.get("train_every") == 1
                     and c.get("train_recent") == 16 and c.get("upgrades") == "attempt,bars,map_sched,objclick")
    # Two checks were rewritten after the first smoke (19 Nov; docs/4-knowledge-planner/kp-bridge-check.md).
    # Training is asked of the baseline runs: Goose trains only once it holds 64 moves of a level, and the
    # planner can clear a level before that. Speed is compared on the click games, whose two runs start
    # together: the last run of a smoke has the GPU to itself, so tu93's two times say nothing; what the
    # dormant planner cost is in its own record.
    side_by_side = [(rows[g, "SOT"], rows[g, "SOT-KP"]) for g in SMOKE_GAMES[:2] if (g, "SOT") in rows and (g, "SOT-KP") in rows]
    slower = max((kp["ms"] / base["ms"] for base, kp in side_by_side), default=float("nan"))
    idle = rows.get(("tu93", "SOT-KP"), {}).get("kp") or {}
    checks = {
        "all six runs played to their cap or to the end of the game": len(rows) == len(jobs),
        "every run has SOT's flags in its run_config.json": all(sot(r["cfg"]) for r in rows.values()),
        "a training step on at least 7 actions in 10, in every SOT run":
            all(r["trained"] >= 0.7 for (_, arm), r in rows.items() if arm == "SOT"),
        "the planner is on in the SOT-KP runs only, with the committed settings":
            all((r["cfg"].get("kp_config_hash") == kp_config_hash()) == (arm == "SOT-KP") for (_, arm), r in rows.items()),
        "no error inside the planner (kp_errors = 0)":
            all(r["kp"].get("kp_errors") == 0 for (_, arm), r in rows.items() if arm == "SOT-KP"),
        f"on ft09 and lp85 an SOT-KP run takes at most 1.5 times as long per action as the SOT run beside it "
        f"(the slowest: {slower:.2f})": slower <= 1.5,
        "on tu93 the planner is dormant and cost nothing (0 s in the reference, no screen parsed)":
            bool(idle) and idle["dormant"] and idle["reference_seconds"] == 0 and idle["parse"]["parsed"] == 0,
    }
    ok = True
    print()
    for name, good in checks.items():
        ok &= bool(good)
        print(f"  {'PASS' if good else 'FAIL'}: {name}")
    for r in read_tsv(out, "failed.tsv"):
        print(f"  failed: {r[2]} {r[0]} s{r[1]}: {r[4]}")
    print("SMOKE: " + ("PASS" if ok else "FAIL") + f"   [{rel(out)}]")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, text in (("plan", cmd_plan, "the arms, their flags, the run count"),
                           ("smoke", cmd_smoke, "six runs and the checks before any sweep"),
                           ("start", cmd_start, "begin the sweep, detached"),
                           ("status", cmd_status, "runs done, runs in progress, time left"),
                           ("pause", cmd_pause, "stop when the runs in progress finish"),
                           ("resume", cmd_resume, "continue after a pause, a crash or a reboot"),
                           ("verdict", cmd_verdict, "the rule and the tables, once every run is in")):
        p = sub.add_parser(name, help=text)
        p.set_defaults(fn=fn)
        if name in ("plan", "smoke", "start"):
            p.add_argument("--jobs", type=int, default=JOBS, help="runs at a time (default 4)")
        if name == "smoke":
            p.add_argument("--evaluate", default="", metavar="FOLDER",
                           help="print the checks again for a smoke already made (no run is started)")
        if name == "start":
            p.add_argument("--new", action="store_true", help="begin another sweep although one is unfinished")
        if name == "verdict":
            p.add_argument("--partial", action="store_true", help="tables for a sweep not yet complete (never a verdict)")
    w = sub.add_parser("_work")         # internal: the detached worker
    w.add_argument("out")
    w.set_defaults(fn=lambda a: work(a.out))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
