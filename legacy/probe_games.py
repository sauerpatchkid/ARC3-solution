"""LEGACY (API path): one-time game-selection tooling. Kept for provenance.

probe_games.py — characterize every public ARC-AGI-3 game from short runs.

Runs the instrumented agent briefly on every available game, then computes a
per-game taxonomy from the transition corpora: action space, change rate,
cells-changed stats, click-locality, levels reached, timing. This is the
empirical input for (a) choosing the Tier 0.5 "trio" games and (b) the Tier 1
dev-set selection meeting.

Usage (from repo root):
    uv run probe_games.py run                 # ~30 min: all games x 1000 actions
    uv run probe_games.py run --games vc33 ft09   # subset (prefix match)
    uv run probe_games.py run --cap 1000 --seed 0
    uv run probe_games.py analyze             # table from the latest probe
    uv run probe_games.py analyze --csv probe_results.csv

`run` writes probe_manifest.json (game -> run dir) so `analyze` knows which
runs/<ts> dirs belong to the probe. Progress prints per game; a failed game is
recorded and skipped, not fatal.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO_ROOT, "probe_manifest.json")
GRID = 64


# ---------------------------------------------------------------- run driver

def fetch_game_list():
    """Game IDs from the ARC API, using the key in the harness .env."""
    import requests
    env_path = os.path.join(REPO_ROOT, "ARC-AGI-3-Agents", ".env")
    api_key = None
    with open(env_path) as f:
        for line in f:
            if line.strip().startswith("ARC_API_KEY"):
                api_key = line.split("=", 1)[1].strip()
    if not api_key:
        sys.exit("ARC_API_KEY not found in ARC-AGI-3-Agents/.env")
    r = requests.get("https://three.arcprize.org/api/games",
                     headers={"X-API-Key": api_key}, timeout=15)
    r.raise_for_status()
    return [g["game_id"] for g in r.json()]


def newest_run_dir(before: set):
    dirs = set(glob.glob(os.path.join(REPO_ROOT, "runs", "*")))
    new = sorted(dirs - before)
    return new[-1] if new else None


def cmd_run(args):
    games = fetch_game_list()
    if args.games:
        games = [g for g in games if any(g.startswith(p) for p in args.games)]
    if not games:
        sys.exit("No games matched.")
    print(f"Probing {len(games)} games x {args.cap} actions (seed {args.seed}).")
    print(f"Rough ETA at ~14 actions/sec: {len(games) * args.cap / 14 / 60:.0f} min\n")

    manifest = {"cap": args.cap, "seed": args.seed,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "games": {}}
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "EVAL_SEED": str(args.seed),
        "EVAL_MAX_ACTIONS": str(args.cap),
        "EVAL_LOG_TRANSITIONS": "1",
        "EVAL_LOG_METRICS": "0",   # keep probe runs lean
        "EVAL_SAVE_VIS": "0",
    })
    for i, game in enumerate(games, 1):
        before = set(glob.glob(os.path.join(REPO_ROOT, "runs", "*")))
        print(f"[{i:2d}/{len(games)}] {game} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            subprocess.run(
                ["uv", "run", "ARC-AGI-3-Agents/main.py",
                 "--agent=action", f"--game={game}"],
                cwd=REPO_ROOT, env=env, timeout=args.timeout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as e:  # keep probing the rest
            status = f"error: {e}"
        run_dir = newest_run_dir(before)
        manifest["games"][game] = {"run_dir": run_dir, "status": status,
                                   "seconds": round(time.time() - t0, 1)}
        print(f"{status} ({time.time() - t0:.0f}s)")
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)  # save progress incrementally
    print(f"\nManifest written: {MANIFEST}\nNext: uv run probe_games.py analyze")


# ------------------------------------------------------------------ analysis

def load_corpus(tdir):
    shards = sorted(glob.glob(os.path.join(tdir, "shard_*.npz")))
    if not shards:
        return None
    data = {}
    for s in shards:
        with np.load(s) as z:
            for k in z.files:
                data.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in data.items()}


def taxonomy(c, cap):
    """Per-game stats from one corpus. Returns a flat dict (one table row)."""
    n = len(c["actions"])
    acts = c["actions"]
    changed = (c["frames"] != c["next_frames"])          # (N, 64, 64) bool
    ch_any = c["changed"].astype(bool)
    row = {}
    row["transitions"] = n
    row["reset_overhead"] = cap - n  # first action + GAME_OVER/RESET losses
    row["levels_reached"] = int(c["levels"].max()) if n else 0

    # action space actually exercised (masking => exercised ~= available)
    discrete_used = sorted({int(a) + 1 for a in acts[acts < 5]})
    n_click = int((acts >= 5).sum())
    row["discrete_actions"] = ",".join(f"A{a}" for a in discrete_used) or "-"
    row["click_pct"] = round(100.0 * n_click / n, 1) if n else 0.0
    kinds = ("click-only" if n_click == n else
             "keyboard-only" if n_click == 0 else "mixed")
    row["action_space"] = kinds

    # change-signal quality
    row["change_rate"] = round(float(ch_any.mean()), 3) if n else 0.0
    per_t = changed.sum(axis=(1, 2))
    ch_idx = np.nonzero(per_t)[0]
    row["cells_changed_med"] = float(np.median(per_t[ch_idx])) if len(ch_idx) else 0.0
    row["cells_changed_p95"] = float(np.percentile(per_t[ch_idx], 95)) if len(ch_idx) else 0.0
    row["cells_ever_changed"] = int((changed.any(axis=0)).sum())

    # click locality (only meaningful where clicks happened AND changed)
    click_mask = acts >= 5
    cc = click_mask & ch_any
    if cc.sum() >= 20:
        idx = np.nonzero(cc)[0]
        clicks = acts[idx] - 5
        ys, xs = clicks // GRID, clicks % GRID
        at_click = changed[idx, ys, xs].mean()
        dists = []
        for j, i2 in enumerate(idx[::max(1, len(idx) // 200)]):
            cy, cx = np.nonzero(changed[i2])
            k = j * max(1, len(idx) // 200)
            if len(cy):
                dists.append(np.min(np.abs(cy - ys[k]) + np.abs(cx - xs[k])))
        row["click_changes_clicked_cell"] = round(float(at_click), 3)
        row["click_to_change_dist_med"] = float(np.median(dists)) if dists else None
    else:
        row["click_changes_clicked_cell"] = None
        row["click_to_change_dist_med"] = None

    row["model_ms_med"] = round(float(np.median(c["model_ms"][1:])), 1) if n > 1 else None
    return row


COLS = ["game", "status", "action_space", "discrete_actions", "click_pct",
        "levels_reached", "change_rate", "cells_changed_med", "cells_changed_p95",
        "cells_ever_changed", "click_changes_clicked_cell",
        "click_to_change_dist_med", "transitions", "reset_overhead", "model_ms_med"]


def cmd_analyze(args):
    with open(MANIFEST) as f:
        manifest = json.load(f)
    cap = manifest["cap"]
    rows = []
    for game, info in manifest["games"].items():
        row = {"game": game, "status": info["status"]}
        tdirs = glob.glob(os.path.join(info["run_dir"] or "", "*", "transitions"))
        c = load_corpus(tdirs[0]) if tdirs else None
        if c is not None and len(c["actions"]):
            row.update(taxonomy(c, cap))
        rows.append(row)

    # pretty print (short columns)
    short = ["game", "action_space", "click_pct", "levels_reached", "change_rate",
             "cells_changed_med", "click_to_change_dist_med", "reset_overhead"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in short}
    print("  ".join(c.ljust(widths[c]) for c in short))
    print("  ".join("-" * widths[c] for c in short))
    for r in sorted(rows, key=lambda r: (r.get("action_space", ""), -(r.get("change_rate") or 0))):
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in short))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows({c: r.get(c, "") for c in COLS} for r in rows)
        print(f"\nFull table -> {args.csv}")

    # trio suggestions
    ok = [r for r in rows if r.get("transitions")]
    def pick(pred, key):
        cands = [r for r in ok if pred(r)]
        return sorted(cands, key=key)[0]["game"] if cands else None
    print("\nTrio candidates (verify by eye / browser before freezing):")
    print("  click-only   :", pick(lambda r: r["action_space"] == "click-only",
                                    lambda r: -r["change_rate"]))
    print("  keyboard-only:", pick(lambda r: r["action_space"] == "keyboard-only",
                                    lambda r: r["change_rate"]))
    print("  mixed        :", pick(lambda r: r["action_space"] == "mixed",
                                    lambda r: abs(0.5 - r["change_rate"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cap", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--timeout", type=int, default=600, help="per-game seconds")
    r.add_argument("--games", nargs="*", help="game id prefixes; default all")
    a = sub.add_parser("analyze")
    a.add_argument("--csv", default="probe_results.csv")
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyze(args)
