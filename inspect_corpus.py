"""Inspect a transition corpus produced by TransitionLogger.

Usage:
    uv run inspect_corpus.py runs/<timestamp>/<game_id>/transitions
    uv run inspect_corpus.py runs/<timestamp>/<game_id>/transitions --verify-seed other/transitions

Prints shard count, transition count, change-rate, per-level counts, and
timing stats. With --verify-seed, checks that two runs took identical action
sequences (the fixed-seeding test).
"""
import argparse
import glob
import os
import sys

import numpy as np


def load_corpus(d):
    shards = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not shards:
        sys.exit(f"No shards found in {d}")
    data = {}
    for s in shards:
        with np.load(s) as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
    return shards, {k: np.concatenate(v) for k, v in data.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("--verify-seed", metavar="OTHER_DIR", default=None,
                    help="second corpus dir; check action sequences match")
    args = ap.parse_args()

    shards, c = load_corpus(args.corpus_dir)
    n = len(c["actions"])
    size_mb = sum(os.path.getsize(s) for s in shards) / 1e6

    print(f"shards: {len(shards)}   transitions: {n}   on disk: {size_mb:.1f} MB "
          f"({1e3 * size_mb / max(n, 1):.2f} KB/transition)")
    print(f"change rate: {c['changed'].mean():.3f}   "
          f"(the class imbalance Phase 2's loss must handle)")

    acts = c["actions"]
    n_click = (acts >= 5).sum()
    print(f"actions: {n - n_click} discrete (ACTION1-5), {n_click} clicks (ACTION6)")
    for a in range(5):
        cnt = (acts == a).sum()
        if cnt:
            chg = c["changed"][acts == a].mean()
            print(f"  ACTION{a+1}: {cnt:>7} taken, {chg:.3f} change rate")
    if n_click:
        print(f"  ACTION6 : {n_click:>7} taken, "
              f"{c['changed'][acts >= 5].mean():.3f} change rate")

    print("per level:")
    for lvl in np.unique(c["levels"]):
        m = c["levels"] == lvl
        print(f"  level {lvl}: {m.sum():>7} transitions, "
              f"change rate {c['changed'][m].mean():.3f}")

    wall, model = c["wall_ms"][1:], c["model_ms"][1:]  # skip first (zeros)
    if len(wall):
        print(f"timing  wall  ms/action: median {np.median(wall):.1f}  p95 {np.percentile(wall, 95):.1f}")
        print(f"timing  model ms/action: median {np.median(model):.1f}  p95 {np.percentile(model, 95):.1f}")
        print(f"  -> model fraction of wall: {np.median(model) / max(np.median(wall), 1e-9):.2f} "
              f"(the rest is mostly server round-trip, absent in the Kaggle sandbox)")

    if args.verify_seed:
        _, c2 = load_corpus(args.verify_seed)
        m = min(len(c["actions"]), len(c2["actions"]))
        same = np.array_equal(c["actions"][:m], c2["actions"][:m])
        print(f"\nseed verification vs {args.verify_seed}:")
        print(f"  first {m} actions identical: {same}")
        if not same:
            first_div = int(np.argmax(c["actions"][:m] != c2["actions"][:m]))
            print(f"  first divergence at action #{first_div}")
            print("  NOTE: identical actions also require the game itself to be "
                  "deterministic; divergence on an animated/stochastic game may "
                  "not indicate a seeding bug — check frames at the divergence point.")


if __name__ == "__main__":
    main()
