"""LEGACY (API path): superseded by run_local.py + compute_metrics.py. Kept for provenance.

summarize_runs.py — comprehensive statistics from evaluation-suite corpora.

Reads suite_manifest.json (from run_suite.py), analyzes every run's transition
corpus, and produces:
  - a per-run table (stdout)
  - per-game aggregates across seeds, mean +/- std (stdout)
  - suite_summary.json: EVERY computed statistic, machine-readable, for
    later analysis / sharing / diffing between algorithm versions
  - suite_summary.csv: the flat per-run table

Usage:
    uv run summarize_runs.py
    uv run summarize_runs.py --manifest suite_manifest.json --out suite_summary

Statistic groups (per run):
  progress      levels reached, action count at each level-up, possible-WIN flag
  budget        transitions, reset overhead, cap utilization
  actions       per-type counts/%, click spatial entropy (early vs late drift)
  signal        raw change rate, indicator cells, MEANINGFUL change rate
                (indicator-corrected), per-action-type change rates,
                cells-changed stats, click locality
  exploration   unique frames (curve at 10/25/50/75/100% of run), unique
                (state,action) pairs, late-run new-state discovery rate,
                exact-repeat fraction
  timing        wall/model ms median+p95, model fraction, est. offline fps
  per-level     transitions, change rates per level segment
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

from metrics_common import find_indicator_cells, GRID, N_CELLS


# ----------------------------------------------------------------- loading

def load_corpus(run_dir):
    tdirs = glob.glob(os.path.join(run_dir or "", "*", "transitions"))
    if not tdirs:
        return None, None
    shards = sorted(glob.glob(os.path.join(tdirs[0], "shard_*.npz")))
    if not shards:
        return None, None
    data = {}
    for s in shards:
        with np.load(s) as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
    corpus = {k: np.concatenate(v) for k, v in data.items()}
    cfg_paths = glob.glob(os.path.join(run_dir, "*", "run_config.json"))
    cfg = json.load(open(cfg_paths[0])) if cfg_paths else {}
    return corpus, cfg


# ----------------------------------------------------------------- helpers

def frame_hashes(frames):
    return np.array([hash(f.tobytes()) for f in frames], dtype=np.int64)


def spatial_entropy(coords, n_cells=N_CELLS):
    """Normalized entropy (0..1) of a click-coordinate distribution.
    1.0 = uniform over the grid; low = concentrated. Uniform-drift detector."""
    if len(coords) < 30:
        return None
    counts = np.bincount(coords, minlength=n_cells).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    return round(float(-(nz * np.log(nz)).sum() / np.log(n_cells)), 4)


def pct(x, n):
    return round(100.0 * x / n, 2) if n else 0.0


# ----------------------------------------------------------------- analysis

def _find_indicator_cells_from_diff(diff):
    """Adapter: compute aggregates from the full diff array, then delegate."""
    n = diff.shape[0]
    freq = diff.mean(axis=0)
    per_t = diff.sum(axis=(1, 2))
    tiny = (per_t > 0) & (per_t <= 2)
    tiny_frac = float(tiny.mean())
    tiny_cell_counts = diff[tiny].sum(axis=0).astype(np.int64) if tiny.any() else np.zeros((GRID, GRID), dtype=np.int64)
    return find_indicator_cells(freq=freq, tiny_frac=tiny_frac,
                                tiny_cell_counts=tiny_cell_counts)


def analyze_run(c, cap):
    n = len(c["actions"])
    acts = c["actions"]
    levels = c["levels"]
    ch_any = c["changed"].astype(bool)
    diff = c["frames"] != c["next_frames"]            # (N,64,64)
    s = {}

    # -- progress
    s["levels_reached"] = int(levels.max())
    lvl_ups = np.nonzero(np.diff(levels) > 0)[0]
    s["actions_at_level_up"] = [int(c["action_nums"][i + 1]) for i in lvl_ups]
    s["transitions"] = n
    s["reset_overhead"] = int(cap - n)
    s["cap_utilization"] = pct(n, cap)
    # ended far below cap without cap-exit => possible WIN (or crash): flag it
    s["possible_win_or_early_exit"] = bool(cap - n > max(0.05 * cap, 200))

    # -- action usage
    for a in range(5):
        s[f"A{a+1}_pct"] = pct(int((acts == a).sum()), n)
    click_mask = acts >= 5
    n_click = int(click_mask.sum())
    s["click_pct"] = pct(n_click, n)
    coords = (acts[click_mask] - 5).astype(int)
    s["click_entropy"] = spatial_entropy(coords)
    third = n // 3
    s["click_entropy_early"] = spatial_entropy((acts[:third][acts[:third] >= 5] - 5).astype(int))
    s["click_entropy_late"] = spatial_entropy((acts[-third:][acts[-third:] >= 5] - 5).astype(int))

    # -- change signal
    s["change_rate_raw"] = round(float(ch_any.mean()), 4)
    indicator = _find_indicator_cells_from_diff(diff)
    s["indicator_cells"] = int(indicator.sum())
    meaningful = diff & ~indicator[None, :, :]
    mf_any = meaningful.any(axis=(1, 2))
    s["change_rate_meaningful"] = round(float(mf_any.mean()), 4)
    per_t = diff.sum(axis=(1, 2))
    chg = per_t[per_t > 0]
    s["cells_changed_med"] = float(np.median(chg)) if len(chg) else 0.0
    s["cells_changed_p95"] = float(np.percentile(chg, 95)) if len(chg) else 0.0
    s["cells_ever_changed"] = int(diff.any(axis=0).sum())
    for a in range(5):
        m = acts == a
        s[f"A{a+1}_change_rate"] = round(float(ch_any[m].mean()), 4) if m.sum() >= 20 else None
        s[f"A{a+1}_meaningful_rate"] = round(float(mf_any[m].mean()), 4) if m.sum() >= 20 else None
    if n_click >= 20:
        s["click_change_rate"] = round(float(ch_any[click_mask].mean()), 4)
        s["click_meaningful_rate"] = round(float(mf_any[click_mask].mean()), 4)
        idx = np.nonzero(click_mask & ch_any)[0]
        if len(idx):
            ys, xs = (acts[idx] - 5) // GRID, (acts[idx] - 5) % GRID
            s["click_changes_clicked_cell"] = round(float(diff[idx, ys, xs].mean()), 4)
            step = max(1, len(idx) // 300)
            dists = []
            for i2 in idx[::step]:
                cy, cx = np.nonzero(diff[i2])
                y0, x0 = (acts[i2] - 5) // GRID, (acts[i2] - 5) % GRID
                dists.append(np.min(np.abs(cy - y0) + np.abs(cx - x0)))
            s["click_to_change_dist_med"] = float(np.median(dists))

    # -- exploration
    fh = frame_hashes(c["frames"])
    seen, curve, uniq = set(), {}, np.zeros(n, dtype=np.int32)
    for i, h in enumerate(fh):
        seen.add(int(h))
        uniq[i] = len(seen)
    for q in (10, 25, 50, 75, 100):
        curve[q] = int(uniq[min(n - 1, n * q // 100)])
    s["unique_frames_curve"] = curve
    s["unique_frames"] = int(uniq[-1])
    s["unique_frames_per_1k_actions"] = round(1000.0 * uniq[-1] / n, 1)
    tail = max(1, n // 10)
    s["late_discovery_rate"] = round(float((uniq[-1] - uniq[-tail]) / tail), 4)
    sa = np.array([hash(f.tobytes() + int(a).to_bytes(4, "little"))
                   for f, a in zip(c["frames"], acts)], dtype=np.int64)
    s["unique_state_actions"] = int(len(np.unique(sa)))
    s["exact_repeat_fraction"] = round(1.0 - s["unique_state_actions"] / n, 4)

    # -- timing
    wall, model = c["wall_ms"][1:], c["model_ms"][1:]
    s["wall_ms_med"] = round(float(np.median(wall)), 1)
    s["wall_ms_p95"] = round(float(np.percentile(wall, 95)), 1)
    s["model_ms_med"] = round(float(np.median(model)), 2)
    s["model_ms_p95"] = round(float(np.percentile(model, 95)), 1)
    s["model_fraction_of_wall"] = round(float(np.median(model) / max(np.median(wall), 1e-9)), 3)
    s["est_offline_actions_per_sec"] = round(1000.0 / max(float(np.mean(model)), 1e-9), 0)

    # -- per-level segments
    s["per_level"] = {}
    for lvl in np.unique(levels):
        m = levels == lvl
        s["per_level"][int(lvl)] = {
            "transitions": int(m.sum()),
            "change_rate_raw": round(float(ch_any[m].mean()), 4),
            "change_rate_meaningful": round(float(mf_any[m].mean()), 4),
        }
    return s


# ----------------------------------------------------------------- reporting

TABLE_COLS = ["game", "seed", "levels_reached", "cap_utilization",
              "change_rate_raw", "change_rate_meaningful", "indicator_cells",
              "unique_frames", "late_discovery_rate", "click_entropy_late",
              "exact_repeat_fraction", "model_ms_med", "wall_ms_med"]

AGG_KEYS = ["levels_reached", "change_rate_meaningful", "unique_frames",
            "unique_frames_per_1k_actions", "late_discovery_rate",
            "exact_repeat_fraction", "click_entropy_late", "model_ms_med"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="suite_manifest.json")
    ap.add_argument("--out", default="suite_summary")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    cap = manifest["cap"]
    results = []
    for r in manifest["runs"]:
        print(f"analyzing {r['game']} seed={r['seed']} ...", flush=True)
        c, cfg = load_corpus(r["run_dir"])
        row = {"game": r["game"], "seed": r["seed"], "status": r["status"],
               "live_seconds": r["seconds"], "run_dir": r["run_dir"],
               "run_config": cfg}
        if c is not None and len(c["actions"]) > 1:
            row.update(analyze_run(c, cap))
        else:
            row["error"] = "no corpus"
        results.append(row)

    # per-run table
    ok = [r for r in results if "error" not in r]
    widths = {c_: max(len(c_), *(len(str(r.get(c_, ""))) for r in ok)) for c_ in TABLE_COLS}
    print()
    print("  ".join(c_.ljust(widths[c_]) for c_ in TABLE_COLS))
    print("  ".join("-" * widths[c_] for c_ in TABLE_COLS))
    for r in ok:
        print("  ".join(str(r.get(c_, "")).ljust(widths[c_]) for c_ in TABLE_COLS))

    # per-game aggregates
    agg = {}
    print("\nPer-game aggregates (mean +/- std over seeds):")
    for game in sorted({r["game"] for r in ok}):
        rows = [r for r in ok if r["game"] == game]
        agg[game] = {}
        parts = []
        for k in AGG_KEYS:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                m, sd = float(np.mean(vals)), float(np.std(vals))
                agg[game][k] = {"mean": round(m, 4), "std": round(sd, 4), "n": len(vals)}
                parts.append(f"{k}={m:.3g}±{sd:.2g}")
        print(f"  {game}: " + "  ".join(parts))

    # outputs
    with open(f"{args.out}.json", "w") as f:
        json.dump({"manifest": {k: v for k, v in manifest.items() if k != "runs"},
                   "runs": results, "aggregates": agg}, f, indent=2, default=str)
    import csv
    flat_cols = [k for k in ok[0] if not isinstance(ok[0][k], (dict, list))] if ok else []
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(ok)
    print(f"\nWrote {args.out}.json (full detail) and {args.out}.csv (flat table)")


if __name__ == "__main__":
    main()
