#!/usr/bin/env python3
"""cpu_check.py - the "off path" check: did a code change leave the agent's moves alone?

Runs are not reproducible on the GPU past the first training step (CLAUDE.md,
"Known measurement caveats"), so a change that should not alter behaviour is
checked on the CPU, where they are: record a short run before the change and
one after it, then compare the two move for move.

    # on the pre-change code:
    uv run python tools/cpu_check.py run --out /tmp/chk/before --game ft09
    # on the changed code:
    uv run python tools/cpu_check.py run --out /tmp/chk/after --game ft09
    uv run python tools/cpu_check.py compare /tmp/chk/before /tmp/chk/after

`run` forces the CPU, deterministic algorithms and a fixed thread count, then
runs run_local.py unchanged on the offline engine. 2,000 actions is about 400
training steps and takes roughly 20 minutes with 4 threads. EVAL_* flags pass
through from the environment, so the adopted agent is checked with

    EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt \\
        uv run python tools/cpu_check.py run --out ... --game tu93

Use the SAME --threads for both runs: the thread count changes the order sums
are taken in, and so the last bits of the weights.

`compare` lines the two recordings up by action number. Every move present in
both must be the same action taken from the same screen and leading to the same
screen. Moves present in only one recording are counted, not failed: a change
to what is LOGGED adds or removes rows without changing what the agent did.
Exit status 1 on any mismatch.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOT = """
import runpy, sys, torch
torch.set_num_threads({threads})
torch.use_deterministic_algorithms(True, warn_only=True)
sys.argv = ["run_local.py", "--game", {game!r}, "--agent", {agent!r}, "--offline"]
runpy.run_path("run_local.py", run_name="__main__")
"""


def run(a):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0",
               OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               EVAL_SEED=str(a.seed), EVAL_MAX_ACTIONS=str(a.actions),
               EVAL_RESULTS_DIR=os.path.abspath(a.out))
    os.makedirs(a.out, exist_ok=True)
    boot = BOOT.format(threads=a.threads, game=a.game, agent=a.agent)
    return subprocess.run([sys.executable, "-c", boot], cwd=ROOT, env=env).returncode


def load(out):
    """{action_num: (action, screen bytes, next screen bytes)} for the one run under `out`."""
    dirs = sorted(glob.glob(os.path.join(out, "runs", "*", "*", "transitions")))
    if len(dirs) != 1:
        sys.exit(f"expected exactly one recorded run under {out}, found {len(dirs)}")
    moves = {}
    for p in sorted(glob.glob(os.path.join(dirs[0], "shard_*.npz"))):
        with np.load(p) as z:
            for n, act, f, nf in zip(z["action_nums"], z["actions"], z["frames"], z["next_frames"]):
                moves[int(n)] = (int(act), f.tobytes(), nf.tobytes())
    return moves


def compare(a):
    A, B = load(a.before), load(a.after)
    common = sorted(set(A) & set(B))
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    bad = [n for n in common if A[n] != B[n]]
    print(f"before: {len(A)} recorded moves   after: {len(B)} recorded moves")
    print(f"compared {len(common)} moves present in both"
          + (f", through action {common[-1]}" if common else ""))
    for name, only in (("before", only_a), ("after", only_b)):
        if only:
            print(f"only in {name}: {len(only)} moves (first few action numbers: {only[:6]})")
    if not common:
        print("RESULT: NOTHING TO COMPARE")
        return 1
    if bad:
        n = bad[0]
        what = ("a different action" if A[n][0] != B[n][0] else
                "a different screen before the move" if A[n][1] != B[n][1] else
                "a different screen after the move")
        print(f"RESULT: MISMATCH on {len(bad)} moves; the first is action {n}: {what} "
              f"(before took action index {A[n][0]}, after took {B[n][0]})")
        return 1
    print("RESULT: IDENTICAL on every move present in both")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="record one short deterministic CPU run")
    r.add_argument("--out", required=True, help="results directory for this run (kept apart from results/)")
    r.add_argument("--game", default="ft09")
    r.add_argument("--agent", default="goose")
    r.add_argument("--actions", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("compare", help="compare two recorded runs move for move")
    c.add_argument("before")
    c.add_argument("after")
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == "run" else compare(a))


if __name__ == "__main__":
    main()
