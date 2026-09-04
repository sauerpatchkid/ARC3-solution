"""scan.py — CLI: corpus statistics for the LLM track.

Answers the three questions week 1 of the plan needs answered before any
labeling budget can be set:

  1. How many transitions are there per game, and how many DISTINCT ones?
     Labeling cost scales with signature count, not transition count, because
     the Judge only ever sees unique signatures (with multiplicity).
  2. What does the decorative mask look like per game, and does it differ from
     what the shared canonicalizer found? (It does — see llm_track/tickers.py.)
  3. How many hindsight anchors exist — level-completing transitions and the
     transitions leading up to them — and are they logged at all? Corpora
     recorded before the level-boundary logging fix are missing the completing
     transition itself.

Usage:
    uv run python -m llm_track.scan --games ft09,ar25,cd82,lp85,ls20
    uv run python -m llm_track.scan --all --limit 40 --out results/llm/scan.json

Read-only: it never writes into a run directory.
"""
import argparse
import collections
import json
import os
import time

import numpy as np

from .corpus import (find_corpora, is_post_fix_boundary, iter_shards,
                     level_events, load_scalars)
from .serializer import serialize, signature
from .tickers import TickerScan

STAGE1 = ["ft09", "ar25", "cd82", "lp85", "ls20"]


def scan_run(corpus_dir, game, limit=None, sample=None):
    """Ticker mask + signature stats + anchor census for one run."""
    t0 = time.time()

    # --- pass 1: decorative mask (needs frames, so this is the expensive one)
    ts = TickerScan()
    for s in iter_shards(corpus_dir, fields=("frames", "next_frames"), limit=limit):
        ts.update(s["frames"], s["next_frames"])
    mask, ticker_info = ts.result()

    # --- anchors: cheap, scalars only, over the WHOLE run
    sc = load_scalars(corpus_dir)
    events = level_events(sc["levels"], sc["action_nums"])
    anchors = [{"action_num": an, "from_level": lo, "to_level": hi,
                "completing_transition_logged": is_post_fix_boundary(sc["action_nums"], i)}
               for i, an, lo, hi in events]
    gaps = np.diff(sc["action_nums"])
    dropped = int((gaps - 1)[gaps > 1].sum())

    # --- pass 2: signatures over a sample
    # Sample from the transitions pass 1 actually visited: with --limit set,
    # only the first `limit` shards are read, so drawing from the full corpus
    # length would silently discard every index past the limit.
    n_total = len(sc["action_nums"])
    n_pop = ts.n
    rng = np.random.default_rng(0)
    take = min(sample or n_pop, n_pop)
    chosen = np.sort(rng.choice(n_pop, take, replace=False)) if take < n_pop else np.arange(n_pop)
    pos = 0
    sigs = collections.Counter()
    buckets = collections.Counter()
    seen = 0
    for s in iter_shards(corpus_dir, limit=limit):
        m = len(s["actions"])
        wanted = chosen[(chosen >= pos) & (chosen < pos + m)] - pos
        for i in wanted:
            rec = serialize(s["frames"][i], s["actions"][i], s["next_frames"][i],
                            mask, level=int(s["levels"][i]),
                            action_num=int(s["action_nums"][i]), game=game)
            sigs[signature(rec)] += 1
            buckets[_bucket(rec)] += 1
            seen += 1
        pos += m
        if limit is not None and pos >= chosen[-1] + 1:
            break

    return {
        "game": game, "corpus_dir": corpus_dir,
        "n_transitions": n_total,
        "n_actions_span": int(sc["action_nums"].max() + 1) if n_total else 0,
        "n_dropped_actions": dropped,
        "raw_change_rate": round(float(sc["changed"].mean()), 4) if n_total else 0.0,
        "ticker": ticker_info,
        "ticker_rows": sorted(set(np.nonzero(mask)[0].tolist())),
        "levelups": len(events),
        "anchors": anchors,
        "signatures_sampled": seen,
        "signatures_unique": len(sigs),
        "dedupe_factor": round(seen / max(len(sigs), 1), 1),
        "buckets": dict(buckets.most_common()),
        "scan_sec": round(time.time() - t0, 1),
    }


def _bucket(rec):
    """Stratification bucket — the pair sampler's unit (design 4.2)."""
    if not rec["changed"]:
        return "unchanged"
    if rec["is_level_completing"]:
        return "level_completing"
    if rec["ticker_only"]:
        return "ticker_only"
    if rec["moves"]:
        return "move"
    if rec["n_changed_nonticker"] <= 8:
        return "small_change"
    return "large_change"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default=",".join(STAGE1),
                    help="comma-separated game ids (default: the Stage-1 set)")
    ap.add_argument("--all", action="store_true", help="every game found under results/runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--limit", type=int, default=40,
                    help="shards per run for the ticker pass (default 40 = 40k transitions)")
    ap.add_argument("--sample", type=int, default=20000,
                    help="transitions to serialize for signature stats (default 20000)")
    ap.add_argument("--out", default=None, help="write JSON here")
    a = ap.parse_args()

    corpora = find_corpora(a.results)
    latest = {}
    for game, d in corpora:                      # keep the newest run per game
        latest[game] = d
    games = sorted(latest) if a.all else [g for g in a.games.split(",") if g in latest]

    rows = []
    for g in games:
        r = scan_run(latest[g], g, limit=a.limit, sample=a.sample)
        rows.append(r)
        t = r["ticker"]
        print(f"\n=== {g} === {r['n_transitions']:,} transitions "
              f"({r['n_dropped_actions']:,} dropped)  [{r['scan_sec']}s]")
        print(f"  ticker      : {t['mask_cells']:>4} cells on rows {r['ticker_rows'][:6]}"
              f"{'...' if len(r['ticker_rows']) > 6 else ''}  (tick_frac {t['tick_frac']})")
        print(f"  signatures  : {r['signatures_unique']:,} unique of {r['signatures_sampled']:,} "
              f"sampled  ->  {r['dedupe_factor']}x dedupe")
        print(f"  buckets     : {r['buckets']}")
        logged = sum(1 for x in r["anchors"] if x["completing_transition_logged"])
        print(f"  anchors     : {r['levelups']} level-ups, {logged} with the completing "
              f"transition logged")

    if rows:
        tot_sig = sum(r["signatures_unique"] for r in rows)
        tot_n = sum(r["n_transitions"] for r in rows)
        print(f"\nTOTAL {len(rows)} games: {tot_n:,} transitions, "
              f"{tot_sig:,} unique signatures in the sampled subset, "
              f"{sum(r['levelups'] for r in rows)} level-ups")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
