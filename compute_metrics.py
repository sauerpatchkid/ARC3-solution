#!/usr/bin/env python3
"""compute_metrics.py - post-run metrics over a transition corpus.

Reads a run's transitions/ folder (.npz shards from eval_common.TransitionLogger)
and computes the Team B metric set:
  1. Level completions + action index at each level-up      (headline)
  2. Unique canonical states, per-action coverage, novelty  (exploration reach)
  3. Meaningful (decorative-corrected) change rate          (doing anything real)
  4. Exact-repeat fraction / redundancy                     (wasted actions)
  5. Action entropy, early vs late window                   (learning signal)
  6. Timing: wall/model ms med+p95, actions/sec             (cost + speed)
  7. Per-1k-action series for 2/3/4                         (stall detection)

Read 2/3/4 TOGETHER, never alone: a high meaningful_change_rate just means the
frame keeps changing, which an agent jiggling a decorative animation achieves at
100%. `unique_states_per_action` is the coverage number; `discovery_auc` is
normalized by final unique count and measures curve SHAPE only.

`novelty_late_per_1k` (new canonical states per 1k actions over the final 10% of
the run) is the leading stall indicator — it flattens thousands of actions before
the level counter confirms the agent is stuck.

Corpus-only, so it can never perturb a run. Writes metrics.json beside the
corpus and optionally appends one row to a shared local_suite.csv.

  python compute_metrics.py results/runs/<ts>/<game>/transitions \
      --game ls20 --agent goose --seed 0 --suite results/local_suite.csv

Indicator-cell detection uses the unified canonicalizer from metrics_common.py
(fixed tickers via DECOR_THRESHOLD + rotating-ticker heuristic), matching
summarize_runs.py so meaningful_change_rate is comparable across pipelines.
EARLY_LATE_FRAC controls the entropy window. numpy-only; two streaming passes;
low memory.
"""
import argparse, csv, glob, hashlib, json, math, os
from collections import Counter
from datetime import datetime, timezone
import numpy as np

from metrics_common import find_indicator_cells, DECOR_THRESHOLD

EARLY_LATE_FRAC = 0.20
WINDOW = 1000            # bucket size for the per-window time series

def shard_paths(d):
    p = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not p: raise SystemExit(f"no shard_*.npz in {d}")
    return p

def iter_shards(paths):
    for p in paths:
        with np.load(p) as z:
            yield {k: z[k] for k in z.files}

def entropy_bits(counter):
    total = sum(counter.values())
    if not total: return 0.0
    return -sum((c/total) * math.log2(c/total) for c in counter.values())

def pass1(paths):
    n, raw_changed = 0, 0
    diff_count = np.zeros((64, 64), dtype=np.int64)
    tiny_count = 0
    tiny_cell_counts = np.zeros((64, 64), dtype=np.int64)
    wall_all, model_all, levelups = [], [], []
    prev_level = None
    for s in iter_shards(paths):
        f, nf = s["frames"], s["next_frames"]
        m = f.shape[0]; n += m
        raw_changed += int(s["changed"].sum())
        diff = (f != nf)
        diff_count += diff.sum(axis=0).astype(np.int64)
        per_t = diff.sum(axis=(1, 2))
        tiny = (per_t > 0) & (per_t <= 2)
        tiny_count += int(tiny.sum())
        if tiny.any():
            tiny_cell_counts += diff[tiny].sum(axis=0).astype(np.int64)
        wall_all.append(s["wall_ms"]); model_all.append(s["model_ms"])
        lv_arr, an_arr = s["levels"], s["action_nums"]
        for i in range(m):
            lv = int(lv_arr[i])
            if prev_level is not None and lv > prev_level:
                levelups.append((int(an_arr[i]), lv))
            prev_level = lv
    wall = np.concatenate(wall_all) if wall_all else np.array([0.0])
    model = np.concatenate(model_all) if model_all else np.array([0.0])
    return {"n": n, "diff_count": diff_count,
            "tiny_count": tiny_count, "tiny_cell_counts": tiny_cell_counts,
            "raw_change_rate": raw_changed/n if n else 0.0,
            "levelups": levelups, "wall": wall, "model": model}

def pass2(paths, mask, n):
    early_n = int(EARLY_LATE_FRAC * n); late_start = n - early_n
    unique, seen_pairs = set(), set()
    repeats = meaningful = auc_running = t = 0
    early_c, late_c = Counter(), Counter()
    # Per-window series. Run-level scalars average away exactly the collapse
    # we're hunting on long runs, so track novelty / change / redundancy in
    # WINDOW-action buckets as well.
    n_buckets = (n + WINDOW - 1) // WINDOW
    b_new = np.zeros(n_buckets, dtype=np.int64)      # newly-seen canonical states
    b_meaningful = np.zeros(n_buckets, dtype=np.int64)
    b_repeats = np.zeros(n_buckets, dtype=np.int64)
    b_n = np.zeros(n_buckets, dtype=np.int64)
    for s in iter_shards(paths):
        f, nf, acts = s["frames"], s["next_frames"], s["actions"]
        for i in range(f.shape[0]):
            b = t // WINDOW
            cf = f[i].copy(); cf[mask] = 0
            h = hashlib.blake2b(cf.tobytes(), digest_size=8).digest()
            before = len(unique)
            unique.add(h)
            if len(unique) != before: b_new[b] += 1
            auc_running += len(unique)
            a = int(acts[i]); pair = (h, a)
            if pair in seen_pairs: repeats += 1; b_repeats[b] += 1
            else: seen_pairs.add(pair)
            if np.any((f[i] != nf[i]) & ~mask): meaningful += 1; b_meaningful[b] += 1
            if t < early_n: early_c[a] += 1
            if t >= late_start: late_c[a] += 1
            b_n[b] += 1
            t += 1
    tot = len(unique)
    # DEPRECATED for cross-run comparison: normalized by FINAL unique-state
    # count, so it measures curve SHAPE, not coverage. A run that finds 172
    # states can score 0.96 while one that finds 184k scores 0.49. Use
    # unique_states_per_action for coverage; never report this one alone.
    auc = (auc_running / (n * tot)) if (n and tot) else 0.0
    e_e, e_l = entropy_bits(early_c), entropy_bits(late_c)
    # Novelty in the final 10% of the run, per 1k actions — the stall detector.
    tail = max(1, int(0.10 * n_buckets))
    tail_new, tail_n = b_new[-tail:].sum(), b_n[-tail:].sum()
    safe = np.maximum(b_n, 1)
    return {"unique_states": tot, "discovery_auc": auc,
            "unique_states_per_action": tot/n if n else 0.0,
            "novelty_late_per_1k": (tail_new/tail_n*WINDOW) if tail_n else 0.0,
            "meaningful_change_rate": meaningful/n if n else 0.0,
            "redundancy": repeats/n if n else 0.0,
            "entropy_early_bits": e_e, "entropy_late_bits": e_l,
            "entropy_delta_bits": e_l - e_e,
            "series": {
                "window": WINDOW,
                "novelty": b_new.tolist(),
                "meaningful": np.round(b_meaningful/safe, 4).tolist(),
                "redundancy": np.round(b_repeats/safe, 4).tolist(),
                "n": b_n.tolist(),
            }}

def compute(corpus_dir):
    paths = shard_paths(corpus_dir); p1 = pass1(paths); n = p1["n"]
    if n == 0: raise SystemExit("corpus is empty")
    freq = p1["diff_count"] / n
    tiny_frac = p1["tiny_count"] / n if n else 0.0
    mask = find_indicator_cells(freq=freq, tiny_frac=tiny_frac,
                                tiny_cell_counts=p1["tiny_cell_counts"])
    p2 = pass2(paths, mask, n)
    wall, model = p1["wall"], p1["model"]
    wall_sec, model_sec = float(wall.sum())/1000.0, float(model.sum())/1000.0
    lus = p1["levelups"]
    return {
        "n_actions": n, "decorative_cells_masked": int(mask.sum()),
        "levels_completed": len(lus),
        "max_level": max((lv for _, lv in lus), default=0),
        "first_levelup_action": lus[0][0] if lus else None,
        "levelup_events": lus,
        "unique_states": p2["unique_states"],
        "discovery_auc": round(p2["discovery_auc"], 4),
        "unique_states_per_action": round(p2["unique_states_per_action"], 6),
        "novelty_late_per_1k": round(p2["novelty_late_per_1k"], 2),
        "raw_change_rate": round(p1["raw_change_rate"], 4),
        "meaningful_change_rate": round(p2["meaningful_change_rate"], 4),
        "redundancy": round(p2["redundancy"], 4),
        "entropy_early_bits": round(p2["entropy_early_bits"], 3),
        "entropy_late_bits": round(p2["entropy_late_bits"], 3),
        "entropy_delta_bits": round(p2["entropy_delta_bits"], 3),
        "wall_ms_med": round(float(np.median(wall)), 2),
        "wall_ms_p95": round(float(np.percentile(wall, 95)), 2),
        "model_ms_med": round(float(np.median(model)), 2),
        "model_ms_p95": round(float(np.percentile(model, 95)), 2),
        "actions_per_sec": round(n/wall_sec, 2) if wall_sec else None,
        "model_bound_aps": round(n/model_sec, 2) if model_sec else None,
        # Per-window series (metrics.json only — too wide for the suite CSV).
        "series": p2["series"],
    }

SUITE_COLUMNS = ["timestamp","game","agent","seed","n_actions","levels_completed",
    "max_level","first_levelup_action","unique_states","discovery_auc",
    "unique_states_per_action","novelty_late_per_1k",
    "raw_change_rate","meaningful_change_rate","redundancy","entropy_early_bits",
    "entropy_late_bits","entropy_delta_bits","wall_ms_med","wall_ms_p95",
    "model_ms_med","model_ms_p95","actions_per_sec","model_bound_aps"]

def _migrate_suite(path):
    """Rewrite an existing suite CSV to the current SUITE_COLUMNS.

    The suite is append-only across schema changes, so when new columns are
    added the old rows must be re-emitted under the new header (missing values
    blank) or every subsequent row would be misaligned against it.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    if header == SUITE_COLUMNS:
        return
    old = [dict(zip(header, r)) for r in rows[1:] if r and r != header]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUITE_COLUMNS)
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in SUITE_COLUMNS})
    print(f"  migrated {path} to the current {len(SUITE_COLUMNS)}-column schema "
          f"({len(old)} existing rows preserved)")

def append_suite(path, m, game, agent, seed):
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "game": game, "agent": agent, "seed": seed,
           **{k: m.get(k) for k in SUITE_COLUMNS if k not in
              ("timestamp","game","agent","seed")}}
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    if exists:
        _migrate_suite(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUITE_COLUMNS)
        if not exists: w.writeheader()
        w.writerow(row)

def print_summary(m, game, agent, seed):
    print(f"\n=== {agent} / {game} / seed {seed} ({m['n_actions']} actions) ===")
    if m["levels_completed"]:
        print(f"  LEVELS COMPLETED: {m['levels_completed']}  (max {m['max_level']}, "
              f"first at action {m['first_levelup_action']})")
    else:
        print("  LEVELS COMPLETED: 0  (censored - none within cap)")
    print(f"  unique states  : {m['unique_states']}  "
          f"({m['unique_states_per_action']} per action, "
          f"{m['decorative_cells_masked']} decorative cells masked)")
    print(f"  novelty (late) : {m['novelty_late_per_1k']} new states / 1k actions "
          f"in the final 10%  [shape AUC {m['discovery_auc']}]")
    print(f"  change rate    : raw {m['raw_change_rate']} | meaningful {m['meaningful_change_rate']}")
    print(f"  redundancy     : {m['redundancy']}")
    print(f"  action entropy : early {m['entropy_early_bits']}b -> late "
          f"{m['entropy_late_bits']}b (delta {m['entropy_delta_bits']}b)")
    print(f"  timing (ms)    : wall {m['wall_ms_med']}/{m['wall_ms_p95']}  "
          f"model {m['model_ms_med']}/{m['model_ms_p95']} (med/p95)")
    print(f"  actions/sec    : {m['actions_per_sec']}  (model-bound ceiling {m['model_bound_aps']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir"); ap.add_argument("--game", default="unknown")
    ap.add_argument("--agent", default="goose"); ap.add_argument("--seed", default="NA")
    ap.add_argument("--out", default=None); ap.add_argument("--suite", default=None)
    a = ap.parse_args()
    m = compute(a.corpus_dir)
    print_summary(m, a.game, a.agent, a.seed)
    out = a.out or os.path.join(os.path.dirname(a.corpus_dir.rstrip("/")), "metrics.json")
    with open(out, "w") as f:
        json.dump({"game": a.game, "agent": a.agent, "seed": a.seed, **m}, f, indent=2)
    print(f"\n  wrote {out}")
    if a.suite:
        append_suite(a.suite, m, a.game, a.agent, a.seed)
        print(f"  appended row to {a.suite}")

if __name__ == "__main__":
    main()
