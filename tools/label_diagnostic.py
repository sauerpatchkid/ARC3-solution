#!/usr/bin/env python3
"""label_diagnostic.py — Plan B step 1: would a novelty label have been more
useful than the frame-change label? Answered on corpora we already have.

    uv run python tools/label_diagnostic.py                 # every run under results/runs
    uv run python tools/label_diagnostic.py --games ft09,ls20 --limit 3   # quick look
    make label-diag

For each recorded run this replays the transition corpus through the ONLINE
canonicalizer (custom_agents/canon.py — the code Plan B would put in the
agent) and reports what fraction of transitions each label calls a success:

  change   raw "did any cell change?"                 (the label Goose trains on today)
  masked   "did any NON-decorative cell change?"      (change with tickers ignored)
  novel    "did we reach a canonical state not seen   (Plan B's label)
            before in this level?"

By construction novel <= masked <= change. The change -> masked gap is what
decoration costs; the masked -> novel gap is what forgetting costs. A label is
only learnable if its positive rate sits away from 1.0, so `change` near 1.0
and `novel` well below it on a game is the case for Plan B on that game.

Also reported, per run: unique canonical states, the end-of-run online mask
size, the offline mask size (compute_metrics.py's, from whole-run aggregates)
and their Jaccard overlap — Plan B §7.1's canonicalizer-agreement check.

Novelty bookkeeping follows the plan: the seen-set is per level (cleared when
the level counter rises, NOT on game over), and the level's first frame counts
as seen. The mask is refreshed on the plan's cadence (warm-up 200, refresh 250)
so keys lag the offline mask slightly, exactly as they would online.

Output: results/diagnostics/label_diagnostic_<stamp>.csv (one row per run) and
.md (per-run rows plus a per-game median table). Nothing here touches the
agent or the scorer.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from canon import OnlineCanonicalizer, DEFAULT_WARMUP, DEFAULT_REFRESH  # noqa: E402
from metrics_common import find_indicator_cells  # noqa: E402

COLUMNS = ["game", "agent", "seed", "reset_on_level", "run", "n", "levels",
           "change", "masked", "novel", "novel_first10", "novel_last10",
           "unique_states", "mask_online", "mask_offline", "mask_jaccard",
           "seconds"]


def find_runs(results_dir):
    for cfg in sorted(glob.glob(os.path.join(results_dir, "runs", "*", "*", "run_config.json"))):
        run_dir = os.path.dirname(cfg)
        shards = sorted(glob.glob(os.path.join(run_dir, "transitions", "shard_*.npz")))
        if not shards:
            continue
        with open(cfg) as f:
            c = json.load(f)
        game = str(c.get("game_id", os.path.basename(run_dir))).split("-")[0]
        yield {"run": os.path.relpath(run_dir, results_dir), "game": game,
               "agent": c.get("agent") or "unknown", "seed": c.get("seed"),
               "reset_on_level": c.get("reset_on_level"), "shards": shards}


def replay(shards, warmup, refresh, chunk=None):
    """One pass over a run. Returns the row dict (without identity fields)."""
    chunk = chunk or refresh
    canon = OnlineCanonicalizer(warmup=warmup, refresh=refresh)
    # Offline aggregates, same as compute_metrics.pass1.
    diff_count = np.zeros((64, 64), dtype=np.int64)
    tiny_count = 0
    tiny_cell_counts = np.zeros((64, 64), dtype=np.int64)

    n = 0
    n_change = n_masked = n_novel = 0
    novel_flags = []                # per transition, for first/last-10% rates
    seen = set()
    all_states = set()
    level = None
    n_levelups = 0

    for path in shards:
        with np.load(path) as z:
            frames, nexts = z["frames"], z["next_frames"]
            changed, levels = z["changed"].astype(bool), z["levels"]
        m = frames.shape[0]
        diff = frames != nexts
        diff_count += diff.sum(axis=0).astype(np.int64)
        per_t = diff.sum(axis=(1, 2))
        tiny = (per_t > 0) & (per_t <= 2)
        tiny_count += int(tiny.sum())
        if tiny.any():
            tiny_cell_counts += diff[tiny].sum(axis=0).astype(np.int64)

        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            # Online: absorb the chunk's transitions (refreshing the mask when
            # due), then key the chunk under the resulting mask.
            canon.update_batch(frames[s:e], nexts[s:e])
            mask = canon.mask
            masked_any = (diff[s:e] & ~mask).any(axis=(1, 2))
            prev_keys = canon.keys_batch(frames[s:e])
            next_keys = canon.keys_batch(nexts[s:e])
            for i in range(e - s):
                lv = int(levels[s + i])
                if level is None:
                    level = lv
                    seen.add(prev_keys[i])
                elif lv > level:              # level-up: fresh memory
                    level = lv
                    n_levelups += 1
                    seen.clear()
                    seen.add(prev_keys[i])
                k = next_keys[i]
                nov = k not in seen
                seen.add(k)
                all_states.add(k)
                n_change += bool(changed[s + i])
                n_masked += bool(masked_any[i])
                n_novel += nov
                novel_flags.append(nov)
        n += m

    if n == 0:
        return None
    online = canon.mask
    freq = diff_count / n
    offline = find_indicator_cells(freq=freq, tiny_frac=tiny_count / n,
                                   tiny_cell_counts=tiny_cell_counts)
    union = int((online | offline).sum())
    inter = int((online & offline).sum())
    tail = max(1, n // 10)
    nf = np.asarray(novel_flags, dtype=bool)
    return {"n": n, "levels": n_levelups,
            "change": n_change / n, "masked": n_masked / n, "novel": n_novel / n,
            "novel_first10": float(nf[:tail].mean()),
            "novel_last10": float(nf[-tail:].mean()),
            "unique_states": len(all_states),
            "mask_online": int(online.sum()), "mask_offline": int(offline.sum()),
            "mask_jaccard": (inter / union) if union else 1.0}


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def per_game_table(rows):
    """Median over runs per (game, agent); k = runs."""
    groups = {}
    for r in rows:
        groups.setdefault((r["game"], r["agent"]), []).append(r)
    out = []
    for (game, agent), rs in sorted(groups.items()):
        med = lambda k: statistics.median(r[k] for r in rs)  # noqa: E731
        out.append({"game": game, "agent": agent, "runs": len(rs),
                    "n_median": int(med("n")),
                    "change": med("change"), "masked": med("masked"),
                    "novel": med("novel"), "novel_last10": med("novel_last10"),
                    "mask_online": int(med("mask_online")),
                    "mask_jaccard": med("mask_jaccard"),
                    "any_levelup": sum(1 for r in rs if r["levels"] > 0)})
    return out


def write_md(path, rows, table, args):
    with open(path, "w") as f:
        f.write(f"# Label diagnostic — {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"{len(rows)} runs replayed from `{args.results}/runs`; "
                f"warm-up {args.warmup}, refresh {args.refresh}.\n\n")
        f.write("Positive-label rate = fraction of transitions the label calls a success. "
                "`change` is the label Goose trains on today; `novel` is Plan B's. "
                "A label near 1.0 has nothing to teach.\n\n")
        f.write("## Per game (median over runs)\n\n")
        f.write("| game | agent | runs | actions | change | masked | novel | novel last 10% | mask cells | mask agree | runs w/ level-up |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for t in table:
            f.write(f"| {t['game']} | {t['agent']} | {t['runs']} | {t['n_median']} | "
                    f"{t['change']:.3f} | {t['masked']:.3f} | {t['novel']:.3f} | "
                    f"{t['novel_last10']:.3f} | {t['mask_online']} | {t['mask_jaccard']:.2f} | "
                    f"{t['any_levelup']}/{t['runs']} |\n")
        f.write("\n`mask agree` = Jaccard overlap of the end-of-run online mask with the "
                "offline scorer's mask (1.00 = identical; both empty also counts as 1.00).\n\n")
        f.write("## Per run\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.environ.get("EVAL_RESULTS_DIR", "results"))
    ap.add_argument("--games", default="", help="comma-separated game ids (default: all)")
    ap.add_argument("--agents", default="", help="comma-separated agents (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="max runs per (game, agent); 0 = all")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--refresh", type=int, default=DEFAULT_REFRESH)
    ap.add_argument("--out", default="", help="output stem (default: results/diagnostics/label_diagnostic_<stamp>)")
    args = ap.parse_args()

    games = set(filter(None, args.games.split(",")))
    agents = set(filter(None, args.agents.split(",")))
    runs = [r for r in find_runs(args.results)
            if (not games or r["game"] in games) and (not agents or r["agent"] in agents)]
    if args.limit:
        kept, count = [], {}
        for r in runs:
            k = (r["game"], r["agent"])
            if count.get(k, 0) < args.limit:
                kept.append(r); count[k] = count.get(k, 0) + 1
        runs = kept
    if not runs:
        sys.exit("no runs matched")

    stem = args.out or os.path.join(args.results, "diagnostics",
                                    f"label_diagnostic_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    csv_path, md_path = stem + ".csv", stem + ".md"

    rows = []
    print(f"{len(runs)} runs -> {csv_path}")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for i, r in enumerate(runs, 1):
            t0 = time.time()
            res = replay(r["shards"], args.warmup, args.refresh)
            if res is None:
                continue
            row = {k: r[k] for k in ("game", "agent", "seed", "reset_on_level", "run")}
            row.update(res)
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            w.writerow(row); f.flush()
            print(f"[{i}/{len(runs)}] {row['game']:<5} {row['agent']:<16} n={row['n']:>7} "
                  f"change={row['change']:.3f} masked={row['masked']:.3f} "
                  f"novel={row['novel']:.3f} mask={row['mask_online']}/{row['mask_offline']} "
                  f"J={row['mask_jaccard']:.2f} ({row['seconds']}s)", flush=True)

    table = per_game_table(rows)
    write_md(md_path, rows, table, args)
    print(f"\nper game (median over runs):")
    print(f"{'game':<5} {'agent':<16} {'runs':>4} {'change':>7} {'masked':>7} {'novel':>7} {'last10':>7} {'mask':>5} {'agree':>6}")
    for t in table:
        print(f"{t['game']:<5} {t['agent']:<16} {t['runs']:>4} {t['change']:>7.3f} {t['masked']:>7.3f} "
              f"{t['novel']:>7.3f} {t['novel_last10']:>7.3f} {t['mask_online']:>5} {t['mask_jaccard']:>6.2f}")
    print(f"\nwrote {csv_path}\n      {md_path}")


if __name__ == "__main__":
    main()
