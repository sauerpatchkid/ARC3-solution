"""LEGACY (API path): superseded by run_local.py + compute_metrics.py. Kept for provenance.

run_suite.py — drive a games x seeds evaluation suite under the eval contract.

Overnight quartet (default): vc33, ft09, g50t, dc22 x seeds 0,1,2 x 25000 actions.

Usage (from repo root, inside tmux for overnight runs):
    uv run run_suite.py                                # the default quartet suite
    uv run run_suite.py --games vc33 ft09 --seeds 0 --cap 2000     # quick variant
    uv run run_suite.py --timeout 7200                 # slower server allowance

Writes suite_manifest.json incrementally (crash/Ctrl-C safe: completed runs
stay recorded). Afterwards:
    uv run summarize_runs.py                           # reads the manifest
"""
import argparse
import glob
import json
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO_ROOT, "suite_manifest.json")

QUARTET = ["vc33", "ft09", "g50t", "dc22"]


def newest_run_dir(before):
    dirs = set(glob.glob(os.path.join(REPO_ROOT, "runs", "*")))
    new = sorted(dirs - before)
    return new[-1] if new else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=QUARTET,
                    help="game id prefixes (default: the quartet)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--cap", type=int, default=25000)
    ap.add_argument("--timeout", type=int, default=5400,
                    help="per-run seconds (default 90 min)")
    ap.add_argument("--agent", default="action")
    args = ap.parse_args()

    jobs = [(g, s) for g in args.games for s in args.seeds]
    eta_h = len(jobs) * args.cap / 14 / 3600
    print(f"Suite: {len(args.games)} games x {len(args.seeds)} seeds x {args.cap} actions "
          f"= {len(jobs)} runs (~{eta_h:.1f} h at 14 fps)\n")

    manifest = {"cap": args.cap, "agent": args.agent, "seeds": args.seeds,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    for i, (game, seed) in enumerate(jobs, 1):
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED": "0", "EVAL_SEED": str(seed),
                    "EVAL_MAX_ACTIONS": str(args.cap),
                    "EVAL_LOG_TRANSITIONS": "1", "EVAL_LOG_METRICS": "1",
                    "EVAL_SAVE_VIS": "0"})
        before = set(glob.glob(os.path.join(REPO_ROOT, "runs", "*")))
        print(f"[{i:2d}/{len(jobs)}] {game} seed={seed} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            subprocess.run(["uv", "run", "ARC-AGI-3-Agents/main.py",
                            f"--agent={args.agent}", f"--game={game}"],
                           cwd=REPO_ROOT, env=env, timeout=args.timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as e:
            status = f"error: {e}"
        secs = round(time.time() - t0, 1)
        manifest["runs"].append({"game": game, "seed": seed, "status": status,
                                 "seconds": secs,
                                 "run_dir": newest_run_dir(before)})
        print(f"{status} ({secs/60:.1f} min)")
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
    print(f"\nDone. Manifest: {MANIFEST}\nNext: uv run summarize_runs.py")


if __name__ == "__main__":
    main()
