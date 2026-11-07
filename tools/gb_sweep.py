#!/usr/bin/env python3
"""gb_sweep.py - Glassbox's online runs: arms x games x seeds of Goose with S5's guide, then the verdict.

    uv run python tools/gb_sweep.py --run g2_smoke --games lp85 --seeds 0 --cap 3000 --jobs 2   # a smoke test
    uv run python tools/gb_sweep.py --run g2 --games dev --seeds 0,1 --book pool_g0b_g1 --g2   # G2 (prereg 5b)
    uv run python tools/gb_sweep.py --run a0_fresh --arms A0-fresh --games dev --seeds 0,1   # A0 run again
    uv run python tools/gb_sweep.py --run dev2 --arms A0-fresh,GB --games dev --seeds 3,4,5,6 \
        --book pool_g0b_g1 --dev2                          # dev sweep 2 (prereg 7a)
    uv run python tools/gb_sweep.py --run g4 --arms A0-fresh,GB --games all --seeds 0,1,2 \
        --book pool_g0b_g1 --g4                            # G4, the confirm (prereg 7)
    uv run python tools/gb_sweep.py ... --dry-run          # the plan and an ETA, nothing run
    uv run python tools/gb_sweep.py ... --resume           # skip runs already in the run's manifest
    uv run python tools/gb_sweep.py --run g4 --status      # runs done, in progress, time left
    uv run python tools/gb_sweep.py --run g4 --pause       # stop when the runs in progress finish
    uv run python tools/gb_sweep.py --run g4 --stop        # stop now: the runs in progress rerun whole

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
loses only the runs in progress (--resume continues).

Pausing (G4 does not run its ~27-30 h in one go). `--pause` (or `touch
<run>/online/STOP`): no new run starts, and the sweep stops once the runs in
progress finish. `--stop`: the sweep and its runs are killed now (its whole
process group; `make` starts it with setsid, so the group holds nothing else).
A run only enters the manifest after run_local.py prints its recording, which it
does only when the game loop ends normally, so a run cut short by a stop, a
crash or a reboot is in no manifest and --resume reruns it whole (its partial
folder stays where it is). `--status`: runs done, runs in progress, time left.
When every run is in, tools/paired_compare.py
writes the verdicts: GB against A0 (--rule dev --expect N, G2's rule), and,
reported, GB-free against A0, GB against GB-free, A0-fresh against A0, and GB and
GB-free against A0-fresh. With --dev2 (dev sweep 2) the verdict is instead GB
against the same-day A0-fresh, under the same rule (verdict_gb_vs_a0_fresh.md).
With --g4 (G4) it is GB against A0-fresh under the confirm rule (--rule confirm
--expect 75), with GB and A0-fresh against A0's 27 Sep runs reported beside it.
Comparisons against A0's 27 Sep runs are skipped when they lack some of the games
and seeds asked (they have seeds 0-2 only), since --expect would block them.

With S5 (GB, GB-free), dev games need --g2 (G2, pre-registration section 5b),
--dev2 (dev sweep 2, section 7a) or --g4 (G4, section 7), each for after its
section is registered. A0-fresh has no S5 and runs on dev games without them.
Seen games (lp85, ls20) run without them, for smoke tests. Held-out games run
only inside G4, and --g4 runs G4 whole or not at all: --arms A0-fresh,GB
--games all --seeds 0,1,2 --cap 100000 (`--games all` is the four tiers in
order: dev, seen, A/B-offline, untouched). Under --g4 a held-out game's levels
are never printed (`done ... levels=sealed`), and every comparison writes the
held-out tiers' per-game rows to a sealed file beside it (paired_compare --seal;
pre-registration section 8). The runs' own folders and logs still hold them.

Copied in pattern from legacy/goose_experiments/upgrade_screen/screen.py (live
code never imports legacy/).
"""
import argparse
import os
import re
import signal
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
ALL = tiers.DEV + tiers.SEEN + tiers.AB_OFFLINE + tiers.UNTOUCHED   # `--games all`: the 25 games, tier by tier
G4 = dict(arms=["A0-fresh", "GB"], games=list(ALL), seeds=["0", "1", "2"], cap=100_000)   # the one G4 run set


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


def verdicts(out, a0, n, n_a0, arms, dev2=False, g4=False):
    """paired_compare's verdicts (module doc). `n` runs per arm; `n_a0` of A0's 27 Sep runs
    match the games and seeds asked (all of them, or the comparisons with A0 are skipped)."""
    m = os.path.join(out, "manifest.tsv")
    dev = ["--rule", "dev", "--expect", str(n)]
    plan = []
    if n_a0 != n:
        print(f"=== A0's 27 Sep runs cover {n_a0} of these {n} game x seed pairs: comparisons with them skipped",
              flush=True)
    if g4:                                      # G4 (prereg 7): GB against the same-day A0, the confirm rule
        confirm = ["--rule", "confirm", "--expect", str(n)]
        plan.append((f"{m}:A0-fresh", f"{m}:GB", confirm, "verdict_gb_vs_a0_fresh.md"))
        if n_a0 == n:                           # reported beside it: A0's 27 Sep runs (112 levels)
            plan.append((f"{a0}:{AGENT}", f"{m}:GB", confirm, "gb_vs_a0.md"))
            plan.append((f"{a0}:{AGENT}", f"{m}:A0-fresh", ["--expect", str(n)], "a0_fresh_vs_a0.md"))
    else:
        if dev2:                                # dev sweep 2 (prereg 7a): GB against the same-day A0
            plan.append((f"{m}:A0-fresh", f"{m}:GB", dev, "verdict_gb_vs_a0_fresh.md"))
        if n_a0 == n:
            if "GB" in arms and not dev2:
                plan.append((f"{a0}:{AGENT}", f"{m}:GB", dev, "verdict_gb_vs_a0.md"))
            if "GB-free" in arms:
                plan.append((f"{a0}:{AGENT}", f"{m}:GB-free", dev, "gb_free_vs_a0.md"))
            if "A0-fresh" in arms:
                plan.append((f"{a0}:{AGENT}", f"{m}:A0-fresh", ["--expect", str(n)], "a0_fresh_vs_a0.md"))
        if {"GB", "GB-free"} <= set(arms):
            plan.append((f"{m}:GB-free", f"{m}:GB", ["--expect", str(n)], "gb_vs_gb_free.md"))
        if "A0-fresh" in arms:
            for arm in ("GB", "GB-free"):
                if arm in arms and not (dev2 and arm == "GB"):
                    plan.append((f"{m}:A0-fresh", f"{m}:{arm}", ["--expect", str(n)],
                                 f"{arm.lower().replace('-', '_')}_vs_a0_fresh.md"))
    for base, new, extra, name in plan:
        if g4:                                  # held-out per-game rows into a sealed file beside the .md
            extra = extra + ["--seal", os.path.join(out, name.replace(".md", ".sealed"))]
        print(f"=== {name}", flush=True)
        subprocess.run([sys.executable, "tools/paired_compare.py", "--base", base, "--new", new, *extra,
                        "--out", os.path.join(out, name)], cwd=ROOT)


def done_line(arm, game, seed, levels, secs, g4=False):
    """The log line for a finished run. Under --g4 a held-out game's levels are not shown."""
    shown = "sealed" if g4 and game in tiers.HELD_OUT else levels
    return f"done {arm:<8} {game} s{seed}  levels={shown}  {secs:5.0f}s"


# ---- pausing: --status, --pause, --stop (module doc) --------------------------------------------------
CONTROL = {"--status", "--pause", "--stop", "--dry-run"}      # calls that only look at or steer a sweep


def proc_args(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
    except OSError:
        return []


def sweep_pids(run):
    """The live processes of a sweep for exactly `run` (uv's wrapper and the python under it),
    found in /proc by their arguments; never a --status, --pause or --stop call."""
    found = []
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == os.getpid():
            continue
        args = proc_args(p)
        if any(x.endswith("gb_sweep.py") for x in args) and not CONTROL & set(args) and "--run" in args \
                and args[args.index("--run") + 1:args.index("--run") + 2] == [run]:
            found.append(int(p))
    return sorted(found)


def group_members(pgid):
    """The live (not zombie) processes of a process group, from /proc."""
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                stat = f.read()
            state, _, pgrp = stat[stat.rindex(")") + 2:].split()[:3]
        except (OSError, ValueError):
            continue
        if int(pgrp) == pgid and state != "Z":
            out.append(int(p))
    return out


def stop(run, wait=30):
    """Stop a sweep now: SIGTERM to its whole process group (the sweep, its runs, their rule
    processes), SIGKILL to whatever is left after `wait` seconds. Refuses unless the sweep leads
    its own group (make starts it with setsid), so nothing else can be hit."""
    pids = sweep_pids(run)
    if not pids:
        print(f"no {run} sweep is running")
        return False
    groups = set()
    for p in pids:
        try:
            groups.add(os.getpgid(p))
        except ProcessLookupError:
            pass
    if not groups <= set(pids):
        sys.exit(f"the {run} sweep does not lead its own process group (it was not started by make), so "
                 f"stopping its group could stop something else: use --pause instead")
    t0 = time.time()
    for g in groups:
        os.killpg(g, signal.SIGTERM)
    while any(group_members(g) for g in groups) and time.time() - t0 < wait:
        time.sleep(0.5)
    for g in groups:
        if group_members(g):
            os.killpg(g, signal.SIGKILL)
    print(f"stopped the {run} sweep and its runs in progress ({time.time() - t0:.0f} s): those runs are in no "
          f"manifest and --resume reruns them whole")
    return True


def pause(run, out):
    """No new run starts; the sweep stops once the runs in progress finish (the STOP file)."""
    if not sweep_pids(run):
        print(f"no {run} sweep is running: nothing to pause")
        return False
    open(os.path.join(out, "STOP"), "w").close()
    print(f"pausing {run}: no new run starts, and the sweep stops when the runs in progress finish (up to "
          f"~50 min at 100,000 actions). --status shows them; continue with --resume")
    return True


def in_progress(log_text):
    """{(arm, game, 's<seed>'): start time} of the runs a sweep's log shows started and not finished
    since its last start-up (a new banner forgets the runs a stop or a reboot cut short)."""
    runs = {}
    for line in log_text.splitlines():
        f = line.split()
        if "GLASSBOX ONLINE" in line:
            runs = {}
        elif len(f) >= 4 and f[0] == "start":
            runs[(f[1], f[2], f[3])] = f[4] if len(f) > 4 else "?"
        elif len(f) >= 4 and f[0] in ("done", "!!"):
            runs.pop((f[1], f[2], f[3].rstrip(":")), None)
    return runs


def status(run, out):
    """Runs done (the manifest), whether the sweep runs or is pausing, its runs in progress and a
    projected time left (from the log: results/glassbox/<run>/<run>_log.txt, as make writes it).
    It shows no levels."""
    done = done_runs(os.path.join(out, "manifest.tsv"))
    log = os.path.join(A.results_dir(), run, f"{run}_log.txt")
    text = open(log).read() if os.path.exists(log) else ""
    plan = re.findall(r"runs to do: (\d+) \(already done: (\d+)\)\s+parallel jobs: (\d+)", text)
    total, jobs = (int(plan[-1][0]) + int(plan[-1][1]), int(plan[-1][2])) if plan else (None, 4)
    secs = [int(x) for x in re.findall(r"^done .*\s(\d+)s$", text, re.M)]
    running = sweep_pids(run)
    print(f"{run}: {len(done)} runs done{f' of {total}' if total else ''}; the sweep is "
          f"{'RUNNING' if running else 'NOT running'}")
    if running and os.path.exists(os.path.join(out, "STOP")):
        print("  pausing: no new run starts; it stops when the runs in progress finish")
    if running:
        for (arm, game, seed), t in sorted(in_progress(text).items(), key=lambda kv: kv[1]):
            print(f"  in progress: {arm:<8} {game} {seed}  since {t}")
    if total and secs and len(done) < total:
        mean = sum(secs) / len(secs)
        print(f"  about {(total - len(done)) * mean / jobs / 3600:.1f} h of running left "
              f"({jobs} at a time; finished runs took {mean / 60:.0f} min on average)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="results/glassbox/<run>/online/ holds everything")
    ap.add_argument("--arms", default="GB,GB-free")
    ap.add_argument("--book", default="", help="the offline run whose book GB carries (EVAL_GB_BOOK)")
    ap.add_argument("--games", default="dev", help="comma-separated, a tier name (dev, seen), or all (G4 only)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--cap", type=int, default=100_000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="skip runs already in the run's manifest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--g2", action="store_true", help="allow dev games: only after prereg section 5b is registered")
    ap.add_argument("--dev2", action="store_true",
                    help="dev sweep 2: allow dev games, verdict GB vs A0-fresh; only after prereg section 7a is registered")
    ap.add_argument("--g4", action="store_true",
                    help="G4, the confirm: allow every game, verdict GB vs A0-fresh under the confirm rule, held-out "
                         "levels sealed; the whole run set only; only after prereg section 7 is registered")
    ap.add_argument("--status", action="store_true", help="the run's sweep: runs done, in progress, time left")
    ap.add_argument("--pause", action="store_true", help="the run's sweep stops once its runs in progress finish")
    ap.add_argument("--stop", action="store_true", help="the run's sweep and its runs in progress are killed now")
    a = ap.parse_args()

    if a.status or a.pause or a.stop:                     # look at or steer a sweep; nothing is run
        out = os.path.join(A.results_dir(), a.run, "online")
        if a.status:
            status(a.run, out)
        elif a.pause:
            pause(a.run, out)
        else:
            stop(a.run)
        return

    if a.games == "all":
        games = list(ALL)
    elif a.games in tiers.TIERS:
        games = list(tiers.TIERS[a.games])
    else:
        games = [g for g in a.games.split(",") if g]
    seeds = [s for s in a.seeds.split(",") if s]
    arms = [x for x in a.arms.split(",") if x]
    for arm in arms:
        arm_env(arm, a.book)
    if "GB" in arms and not a.book:
        sys.exit("arm GB needs --book <offline run>")
    if a.book and not os.path.isdir(os.path.join(A.results_dir(), a.book)):
        sys.exit(f"--book: no run {a.book} under {A.results_dir()}")
    if a.g4 and (a.g2 or a.dev2):
        sys.exit("--g4 is a run of its own: not with --g2 or --dev2")
    if a.g4 and dict(arms=arms, games=games, seeds=seeds, cap=a.cap) != G4:
        sys.exit("--g4 runs G4 whole or not at all (pre-registration section 7): "
                 "--arms A0-fresh,GB --games all --seeds 0,1,2 --cap 100000")
    held = [g for g in games if g in tiers.HELD_OUT]
    if held and not a.g4:
        sys.exit(f"{', '.join(held)}: held-out games run only inside G4 (--g4, pre-registration section 7, "
                 "for after it is registered)")
    dev = [g for g in games if g in tiers.DEV]
    if dev and not (a.g2 or a.dev2 or a.g4) and {"GB", "GB-free"} & set(arms):
        sys.exit(f"{', '.join(dev)}: with S5, dev games are G2 (--g2, pre-registration section 5b), dev sweep 2 "
                 "(--dev2, section 7a) or G4 (--g4, section 7), each for after its section is registered")
    if a.dev2 and not {"GB", "A0-fresh"} <= set(arms):
        sys.exit("--dev2 compares GB with the same-day A0: --arms needs GB and A0-fresh")

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
        print(done_line(arm, g, s, levels, time.time() - t0, a.g4), flush=True)

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
    verdicts(out, a0, len(games) * len(seeds), n_a0, arms, a.dev2, a.g4)   # --expect blocks if an arm lacks a run


if __name__ == "__main__":
    main()
