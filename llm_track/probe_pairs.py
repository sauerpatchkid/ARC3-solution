"""probe_pairs.py — build the frozen pair set for Probe A (judge validity).

Probe A asks one question before anything is built on top of it: can a small
LLM tell progress from noise on ARC-AGI-3 transitions? This script builds the
test set, llm_track/judge.py asks the models, and llm_track/probe_report.py
scores them. The pair set is built ONCE and shared by every model, so the 4B
and the 9B are graded on identical questions.

THREE TIERS
  sanity_ticker    a move that changed real cells  vs  a move where only the
                   ticker (timer / move counter) changed.   Expected: real move.
  sanity_nochange  a move that changed real cells  vs  a move that changed
                   nothing.                                 Expected: real move.
  anchor           one of the last few real moves before a level was completed
                   vs  a real move from much earlier in the same level.
                                                            Expected: the anchor.

The sanity tiers only test whether a model understands the text format. A
one-line rule ("more real cells changed wins") gets them right, so passing them
proves nothing about judgment. The anchor tier is the real test.

WHAT "ANCHOR" MEANS, AND ITS LIMITS
  The only ground truth about progress in this corpus is where levels were
  completed. Moves just before a completion are ENRICHED for progress, not
  guaranteed to be progress: Goose explores stochastically, so some of them
  are noise. A perfect judge therefore scores well below 100% on this tier.
  What matters is whether it beats chance, and beats the size baseline the
  report computes on the same pairs.

  The level-COMPLETING move itself is deliberately excluded. Its next frame is
  the new level's first screen, so it "changes" most of the grid and any rule
  preferring bigger changes wins it for free. (All 73 level-ups in the current
  corpus predate the logging fix, so none are logged anyway.)

  Built from Goose's own runs only (run_config.json agent == stochastic_goose).
  Each game's ticker mask comes from its LARGEST Goose run, never its newest —
  the newest ft09 run was a 2,000-action random-agent smoke test.

  Both sides of an anchor pair change real, non-ticker cells, so the judge
  cannot win on "something happened vs nothing happened". Pairs whose two texts
  are identical are dropped and counted: no judge can separate them, and how
  often that happens is itself a measurement of the serializer.

NO HINDSIGHT LEAKS INTO THE PROMPT
  Every transition is serialized with is_level_completing=False, and the
  builder asserts the completion marker never appears in any text.

THE UNIT OF EVIDENCE IS THE LEVEL-UP EVENT, NOT THE PAIR
  Each event contributes up to 5 anchors x 2 controls, which are correlated.
  The report bootstraps over events. 56 of the 73 events are ft09, so read the
  per-game breakdown before reading the pooled number.

Usage:
    uv run python -m llm_track.probe_pairs          # -> results/llm/probeA/
"""
import argparse
import collections
import json
import os
import subprocess
import time

import numpy as np

from .corpus import (GOOSE, CorpusReader, find_corpora, game_of,
                     is_post_fix_boundary, largest_per_game, level_events)
from .scan import STAGE1, _bucket
from .serializer import SCHEMA_VERSION, serialize
from .tickers import TickerScan

K_ANCHORS = 5            # real moves taken from just before each level-up
LOOKBACK = 60            # how far back (transitions) to look for them
CONTROLS_PER_ANCHOR = 2
MIN_GAP = 200            # controls come from at least this many moves earlier
MIN_GAP_SHORT = 20       # fallback for short levels
MIN_POOL = 10            # below this many candidate controls, skip the event
SANITY_PER_GAME = 40     # pairs per sanity tier per game
MASK_SHARDS = 50         # fixed, recorded window for each game's ticker mask
LEAK_MARKER = "COMPLETED THE LEVEL"


def build_masks(corpora):
    """One decorative mask per game, from its LARGEST Goose run, over a fixed
    window (see tickers.scan_corpus for why the window must be fixed)."""
    sources = largest_per_game(corpora)
    masks, info = {}, {}
    for g, d in sorted(sources.items()):
        ts = TickerScan()
        r = CorpusReader(d)
        for s in range(min(MASK_SHARDS, r.n_shards)):
            sh = r.shard(s)
            ts.update(sh["frames"], sh["next_frames"])
        masks[g], stats = ts.result()
        info[g] = {"source_run": d, "window_shards": MASK_SHARDS, **stats}
    return masks, info, sources


def side(reader, i, mask, game):
    """Serialize transition i with no hindsight, and keep the features the
    report's baselines need."""
    t = reader.get(i)
    rec = serialize(t["frame"], t["action"], t["next_frame"], mask,
                    level=t["level"], action_num=t["action_num"], game=game,
                    is_level_completing=False, level_delta=0)
    assert LEAK_MARKER not in rec["text"], "hindsight leaked into a prompt"
    return {
        "text": rec["text"],
        "action_num": t["action_num"],
        "n_changed": rec["n_changed"],
        "n_changed_nonticker": rec["n_changed_nonticker"],
        "n_components": rec["n_components"],
        "n_moves": len(rec["moves"]),
        "ticker_only": rec["ticker_only"],
        "bucket": _bucket(rec),
    }


def segment_start(levels, i):
    """First index of the contiguous same-level run that ends at i."""
    lv = levels[i]
    j = i
    while j > 0 and levels[j - 1] == lv:
        j -= 1
    return j


def anchor_pairs(corpora, masks, rng, stats):
    pairs = []
    for name, d in corpora:
        g = game_of(name)
        r = CorpusReader(d)
        sc = r.scalars
        run_id = os.path.basename(os.path.dirname(os.path.dirname(d)))
        for i, an, lo, hi in level_events(sc["levels"], sc["action_nums"]):
            ev = f"{run_id}/{name}/{an}"
            # Post-fix corpora log the completing move at index i; skip it (see
            # module docstring). Pre-fix corpora never logged it.
            start = i - 1 if is_post_fix_boundary(sc["action_nums"], i) else i
            seg0 = segment_start(sc["levels"], start)

            anchors = []
            for j in range(start, max(seg0, start - LOOKBACK) - 1, -1):
                s = side(r, j, masks[g], g)
                if s["n_changed_nonticker"] > 0:
                    s["steps_before_levelup"] = int(an) - s["action_num"] + 1
                    anchors.append(s)
                    if len(anchors) == K_ANCHORS:
                        break
            if not anchors:
                stats["skipped_no_real_anchor"].append(ev)
                continue

            first_anchor = min(a["action_num"] for a in anchors)
            idx_of = {int(x): k for k, x in enumerate(sc["action_nums"][seg0:start + 1], seg0)}
            cut = idx_of.get(first_anchor, start) - MIN_GAP
            if cut - seg0 < MIN_POOL:
                cut = idx_of.get(first_anchor, start) - MIN_GAP_SHORT
            if cut - seg0 < MIN_POOL:
                stats["skipped_level_too_short"].append(ev)
                continue

            need = CONTROLS_PER_ANCHOR * len(anchors)
            cand = rng.choice(np.arange(seg0, cut), size=min(cut - seg0, need * 6),
                              replace=False)
            controls = []
            for j in np.sort(cand):
                s = side(r, int(j), masks[g], g)
                if s["n_changed_nonticker"] > 0:
                    controls.append(s)
            rng.shuffle(controls)
            controls = controls[:need]
            if not controls:
                stats["skipped_no_real_control"].append(ev)
                continue

            stats["events_used"].append(ev)
            for k, c in enumerate(controls):
                a = anchors[k % len(anchors)]
                if a["text"] == c["text"]:
                    stats["identical_text_dropped"] += 1
                    continue
                pairs.append({
                    "pair_id": f"anc-{g}-{run_id}-{an}-a{k % len(anchors)}-c{k}",
                    "tier": "anchor", "game": g, "level": int(lo), "event": ev,
                    "x": a, "y": c, "truth": "x",
                })
    return pairs


def sanity_pairs(sources, masks, rng, stats):
    pairs = []
    for g in STAGE1:
        if g not in sources:
            continue
        r = CorpusReader(sources[g])
        idx = np.sort(rng.choice(len(r), size=min(len(r), 3000), replace=False))
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in idx:
            s = side(r, int(i), masks[g], g)
            kind = ("real" if s["n_changed_nonticker"] > 0 else
                    "ticker" if s["ticker_only"] else
                    "none" if s["n_changed"] == 0 else "other")
            by[r.scalars["levels"][i]][kind].append(s)
        for tier, other in (("sanity_ticker", "ticker"), ("sanity_nochange", "none")):
            made = 0
            for lv, kinds in by.items():
                reals, others = kinds["real"], kinds[other]
                for a, b in zip(reals, others):
                    if made >= SANITY_PER_GAME:
                        break
                    pairs.append({
                        "pair_id": f"snt-{tier[7:]}-{g}-{made:03d}",
                        "tier": tier, "game": g, "level": int(lv),
                        "event": f"{g}/sanity", "x": a, "y": b, "truth": "x",
                    })
                    made += 1
            stats[f"{tier}_per_game"][g] = made
    return pairs


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/llm/probeA")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    # Goose's own play only: a teammate's agent or the random floor behaves
    # differently, and mixing them would change what "a move" looks like.
    corpora = find_corpora(a.results, agent=GOOSE)
    print(f"[probe_pairs] {len(corpora)} Goose corpora found")

    masks, mask_info, sources = build_masks(corpora)
    print(f"[probe_pairs] masks built for {len(masks)} games "
          f"({time.time() - t0:.0f}s)")

    stats = {"events_used": [], "skipped_no_real_anchor": [],
             "skipped_level_too_short": [], "skipped_no_real_control": [],
             "identical_text_dropped": 0,
             "sanity_ticker_per_game": {}, "sanity_nochange_per_game": {}}
    pairs = anchor_pairs(corpora, masks, rng, stats)
    pairs += sanity_pairs(sources, masks, rng, stats)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    tiers = collections.Counter(p["tier"] for p in pairs)
    per_game = collections.Counter(p["game"] for p in pairs if p["tier"] == "anchor")
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "schema_version": SCHEMA_VERSION,
        "seed": a.seed,
        "params": {"K_ANCHORS": K_ANCHORS, "LOOKBACK": LOOKBACK,
                   "CONTROLS_PER_ANCHOR": CONTROLS_PER_ANCHOR, "MIN_GAP": MIN_GAP,
                   "MIN_GAP_SHORT": MIN_GAP_SHORT, "SANITY_PER_GAME": SANITY_PER_GAME,
                   "MASK_SHARDS": MASK_SHARDS},
        "n_pairs": len(pairs), "tiers": dict(tiers),
        "anchor_pairs_per_game": dict(per_game),
        "masks": {g: {k: v for k, v in i.items()} for g, i in mask_info.items()},
        **{k: (len(v) if isinstance(v, list) else v) for k, v in stats.items()},
        "events": stats["events_used"],
        "skipped": {k: stats[k] for k in stats if k.startswith("skipped")},
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[probe_pairs] {len(pairs)} pairs: {dict(tiers)}")
    print(f"[probe_pairs] anchor events used: {len(stats['events_used'])}; "
          f"skipped: " + ", ".join(f"{k[8:]}={len(stats[k])}" for k in stats
                                   if k.startswith("skipped")))
    print(f"[probe_pairs] identical-text pairs dropped: {stats['identical_text_dropped']}")
    print(f"[probe_pairs] anchor pairs per game: {dict(per_game.most_common())}")
    print(f"[probe_pairs] wrote {a.out}/pairs.jsonl and meta.json "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
