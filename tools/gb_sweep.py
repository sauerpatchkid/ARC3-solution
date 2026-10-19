#!/usr/bin/env python3
"""gb_sweep.py - Glassbox's online runs: arms x games x seeds of Goose with S5's guide, then the verdict.

    uv run python tools/gb_sweep.py --run g2_smoke --games lp85 --seeds 0 --cap 3000 --jobs 2   # a smoke test
    uv run python tools/gb_sweep.py --run g2 --games dev --seeds 0,1 --book pool_g0b_g1 --g2   # G2 (prereg 5b)
    uv run python tools/gb_sweep.py --run a0_fresh --arms A0-fresh --games dev --seeds 0,1   # A0 run again
    uv run python tools/gb_sweep.py ... --dry-run          # the plan and an ETA, nothing run
    uv run python tools/gb_sweep.py ... --resume           # skip runs already in the run's manifest

Arms (all `--agent mb_gated_att`, the adopted Goose, so A0 is the reference):
  GB       EVAL_GB=1 EVAL_GB_BOOK=<--book>: S5 with the book's no-op rules
  GB-free  EVAL_GB=1 and no book: colour no-ops and goal hints, no LLM anywhere
  A0-fresh EVAL_GB unset: the adopted Goose itself, run again today. A0's runs are
           from 27 Sep, so this is the control for drift since then and for seed noise
A0's own runs are rows of A0's confirm manifest ([run] manifest in glassbox.toml),
never re-run. The rows for exactly the games and seeds asked are copied into
<run>/online/a0.manifest.tsv (paths unchanged; nothing is moved), so
`paired_compare --expect N` sees N runs on each side. No arm inherits the shell's
EVAL_GB or EVAL_GB_BOOK.

Runs go 4 at a time by default (the GPU is shared: 4 runs total ~162 act/s against
140 for one; CLAUDE.md). Each run gets its own results folder
(<run>/online/runs/<arm>/<game>_s<seed>/), is scored by compute_metrics.py and
appended to <run>/online/manifest.tsv as soon as it finishes, so a pause or crash
loses only the runs in progress (--resume continues). `touch <run>/online/STOP`
pauses after the runs in progress. When every run is in, tools/paired_compare.py
writes the verdicts: GB against A0 (--rule dev --expect N, G2's rule), and,
reported, GB-free against A0, GB against GB-free, A0-fresh against A0, and GB and
GB-free against A0-fresh.

With S5 (GB, GB-free), dev games are G2 (pre-registration section 5b): they need
--g2, which is for after that section is registered. A0-fresh has no S5 and runs
on dev games without it. Seen games (lp85, ls20) run without it, for smoke tests.
Held-out games are refused.

Copied in pattern from legacy/goose_experiments/upgrade_screen/screen.py (live
code never imports legacy/).
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

import tiers  # noqa: E402
from gb import artifacts as A  # noqa: E402
from manifest import read_manifest  # noqa: E402

AGENT = "mb_gated_att"
SPEED = {1: 140, 2: 155, 3: 158, 4: 162}     # total act/s across parallel runs, measured (screen.py)
TRANSITIONS = re.compile(r"^\[run_local\] transitions: (.+)$", re.M)


def arm_env(arm, book):
    if arm == "GB":       # absolute: each run has its own EVAL_RESULTS_DIR, which a relative name would follow
        return {"EVAL_GB": "1", "EVAL_GB_BOOK": os.path.join(A.results_dir(), book)}
    if arm == "GB-free":
        return {"EVAL_GB": "1"}
    if arm == "A0-fresh":
        return {}
    raise SystemExit(f"unknown arm {arm!r}; arms are GB, GB-free, A0-fresh")


def run_env(arm, book, base):
    """The shell's environment without any EVAL_GB*, plus the arm's: nothing leaks into an arm."""
    env = {k: v for k, v in base.items() if not k.startswith("EVAL_GB")}
    env.update(arm_env(arm, book))
    return env


def done_runs(path):
    return {(r["game"], r["seed"], r["arm"]) for r in read_manifest(path)} if os.path.exists(path) else set()


def write_a0(out, games, seeds):
    """A0's confirm rows for exactly these games and seeds, as their own manifest."""
    cfg = A.load_config()["run"]
    rows = [r for r in read_manifest(os.path.join(ROOT, cfg["manifest"]), strict=True)
            if r["game"] in games and r["seed"] in seeds and r["arm"] == AGENT]
    path = os.path.join(out, "a0.manifest.tsv")
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{r['run_dir']}\t{r['game']}\t{r['seed']}\t{r['arm']}\n")
    return path, len(rows)


def verdicts(out, a0, n, arms):
    m = os.path.join(out, "manifest.tsv")
    plan = []
    if "GB" in arms:
        plan.append((f"{a0}:{AGENT}", f"{m}:GB", ["--rule", "dev", "--expect", str(n)], "verdict_gb_vs_a0.md"))
    if "GB-free" in arms:
        plan.append((f"{a0}:{AGENT}", f"{m}:GB-free", ["--rule", "dev", "--expect", str(n)], "gb_free_vs_a0.md"))
    if {"GB", "GB-free"} <= set(arms):
        plan.append((f"{m}:GB-free", f"{m}:GB", ["--expect", str(n)], "gb_vs_gb_free.md"))
    if "A0-fresh" in arms:
        plan.append((f"{a0}:{AGENT}", f"{m}:A0-fresh", ["--expect", str(n)], "a0_fresh_vs_a0.md"))
        for arm in ("GB", "GB-free"):
            if arm in arms:
                plan.append((f"{m}:A0-fresh", f"{m}:{arm}", ["--expect", str(n)],
                             f"{arm.lower().replace('-', '_')}_vs_a0_fresh.md"))
    for base, new, extra, name in plan:
        print(f"=== {name}", flush=True)
        subprocess.run([sys.executable, "tools/paired_compare.py", "--base", base, "--new", new, *extra,
                        "--out", os.path.join(out, name)], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="results/glassbox/<run>/online/ holds everything")
    ap.add_argument("--arms", default="GB,GB-free")
    ap.add_argument("--book", default="", help="the offline run whose book GB carries (EVAL_GB_BOOK)")
    ap.add_argument("--games", default="dev", help="comma-separated, or a tier name (dev, seen)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--cap", type=int, default=100_000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="skip runs already in the run's manifest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--g2", action="store_true", help="allow dev games: only after prereg section 5b is registered")
    a = ap.parse_args()

    games = list(tiers.TIERS[a.games]) if a.games in tiers.TIERS else [g for g in a.games.split(",") if g]
    seeds = [s for s in a.seeds.split(",") if s]
    arms = [x for x in a.arms.split(",") if x]
    for arm in arms:
        arm_env(arm, a.book)
    if "GB" in arms and not a.book:
        sys.exit("arm GB needs --book <offline run>")
    if a.book and not os.path.isdir(os.path.join(A.results_dir(), a.book)):
        sys.exit(f"--book: no run {a.book} under {A.results_dir()}")
    held = [g for g in games if g in tiers.HELD_OUT]
    if held:
        sys.exit(f"{', '.join(held)}: held-out games are never run here")
    dev = [g for g in games if g in tiers.DEV]
    if dev and not a.g2 and {"GB", "GB-free"} & set(arms):
        sys.exit(f"{', '.join(dev)}: dev games are G2 (pre-registration section 5b); they need --g2, which is "
                 "for after that section is registered")

    out = os.path.join(A.results_dir(), a.run, "online")
    manifest = os.path.join(out, "manifest.tsv")
    if os.path.exists(manifest) and not a.resume and not a.dry_run:
        sys.exit(f"{manifest} exists: use --resume to continue it, or another --run")
    done = done_runs(manifest)
    jobs = [(g, s, arm) for s in seeds for g in games for arm in arms if (g, s, arm) not in done]
    speed = SPEED.get(a.jobs, 155)
    eta_h = (len(jobs) * a.cap / speed + len(jobs) * 60 / max(a.jobs, 1)) / 3600
    print("=" * 72)
    print(f" GLASSBOX ONLINE  {len(arms)} arms ({', '.join(arms)}) x {len(games)} games x {len(seeds)} seeds "
          f"x {a.cap:,} actions; book {a.book or '-'}")
    print(f" runs to do: {len(jobs)} (already done: {len(done)})   parallel jobs: {a.jobs}   "
          f"ETA ~{eta_h:.1f} h (~{speed} act/s total; S5 adds re-binding pauses)")
    print(f" output: {out}")
    print("=" * 72)
    if a.dry_run:
        print("DRY RUN - nothing executed.")
        return

    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    a0, n_a0 = write_a0(out, games, seeds)
    stop_file = os.path.join(out, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    lock = threading.Lock()

    def run(job):
        g, s, arm = job
        env = run_env(arm, a.book, os.environ)
        env.update(EVAL_SEED=str(s), EVAL_MAX_ACTIONS=str(a.cap), PYTHONHASHSEED="0",
                   EVAL_RESULTS_DIR=os.path.join(out, "runs", arm, f"{g}_s{s}"))
        log = os.path.join(out, "logs", f"{arm}_{g}_s{s}.log")
        t0 = time.time()
        with open(log, "w") as f:
            subprocess.run([sys.executable, "run_local.py", "--game", g, "--agent", AGENT], cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
        text = open(log).read()
        m = TRANSITIONS.search(text)
        if not m:
            print(f"!! {arm} {g} s{s}: no recording (see {log})", flush=True)
            return
        corpus = m.group(1).strip()
        label = AGENT if arm == "A0-fresh" else f"gb_{arm}"
        subprocess.run([sys.executable, "compute_metrics.py", corpus, "--game", g, "--agent", label,
                        "--seed", str(s)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        rundir = os.path.dirname(corpus)
        if not os.path.exists(os.path.join(rundir if os.path.isabs(rundir) else os.path.join(ROOT, rundir),
                                           "metrics.json")):
            print(f"!! {arm} {g} s{s}: scoring failed", flush=True)
            return
        levels = len(re.findall(r"Score changed from [0-9]", text))
        with lock:
            with open(manifest, "a") as f:
                f.write(f"{rundir}\t{g}\t{s}\t{arm}\n")
        print(f"done {arm:<8} {g} s{s}  levels={levels}  {time.time() - t0:5.0f}s", flush=True)

    running, stopped = [], False
    for job in jobs:
        while len([t for t in running if t.is_alive()]) >= a.jobs:
            time.sleep(2)
        if os.path.exists(stop_file):
            stopped = True
            break
        t = threading.Thread(target=run, args=(job,))
        t.start()
        running.append(t)
        print(f"start {job[2]:<8} {job[0]} s{job[1]}  {datetime.now():%H:%M:%S}", flush=True)
        time.sleep(1.5)                                   # stagger GPU start-up
    for t in running:
        t.join()
    if stopped:
        print("=== PAUSED. Continue with the same command and --resume", flush=True)
        return
    missing = [j for j in jobs if j not in done_runs(manifest)]
    if missing:
        print(f"=== {len(missing)} runs missing (see logs); rerun with --resume before the verdict", flush=True)
        return
    verdicts(out, a0, len(games) * len(seeds), arms)       # --expect blocks if A0 lacks a run


if __name__ == "__main__":
    main()
