"""s0_mask.py - S0 MASK: run each bar in the decoration mask out to the screen's edges.

The problem it fixes: the decoration mask (gb/core/evidence.py's ticker scan)
finds a bar from the cells that change often, so it misses a bar's last cells,
which change only once per attempt and late. On m0r0 the move-budget bar runs
along row 63 and the mask stops at column 57: when cell (63, 58) empties at
move 137 of an attempt, the same masked screen ends two ways, and every rule
that applies there is refused trust ("seen to end in more than one way"). G0
found this: all 11 of m0r0's near-miss rules miss trust for that reason alone
(docs/3-glassbox/g0-guard-rescue.md).

What it does. A BAR is a run of at least `bar_min_length` masked cells along a
row within `band` cells of the top or bottom edge, or along a column within
`band` cells of the left or right edge (the band where the ticker scan keeps
bars: gridtools.BAR_BORDER). With `complete_bars`, every such row (or column)
is masked in full. Rows and columns are judged on the scan's own mask, so a bar
along one edge never extends a bar along another. Nothing else changes: the
same moves, the same levels, the same splits as CP1 (the training level; the
transfer level's first `fit_moves` moves and the rest), built the way
LevelEvidence.from_corpus and LevelEvidence.split build them. With
`complete_bars = false` the evidence is CP1's exactly (tests/test_gb_mask.py).

Masking is an inference: a masked cell is never predicted or checked. A full
row masks 6 more cells on m0r0 and 4 on tu93; the report lists every game's
count, so a mask that swallowed a real playing cell would show.

Reads  the recording (a run's transitions/ folder) and the levels to build
Writes <run>/evidence/<game>_L<k>_train.npz, <game>_L<j>_{fit,score}.npz
       <run>/s0_mask/<game>.json    per split: mask cells, cases and conflicting
                                    cases with the scan's mask and with S0's
"""
import json
import os

import numpy as np

from gb import artifacts as A
from gb.core.evidence import MASK_MOVES, LevelEvidence, read_moves, ticker_mask
from gridtools import BAR_BORDER, GRID

VERSION = 1                     # bump when what the output means changes (artifacts.py)
DEFAULTS = {"complete_bars": True, "bar_min_length": 8, "band": BAR_BORDER, "fit_moves": 300}


def complete_bars(mask, settings=DEFAULTS):
    """The mask with each bar run to the screen's edges (module doc)."""
    if not settings["complete_bars"]:
        return mask
    out = mask.copy()
    band, n = settings["band"], settings["bar_min_length"]
    edge = list(range(band)) + list(range(GRID - band, GRID))

    def longest(line):
        run = best = 0
        for v in line:
            run = run + 1 if v else 0
            best = max(best, run)
        return best
    for y in edge:
        if longest(mask[y, :]) >= n:
            out[y, :] = True
    for x in edge:
        if longest(mask[:, x]) >= n:
            out[:, x] = True
    return out


def pick_levels(game, corpus, min_train_moves=500, min_transfer_cases=50):
    """CP1's levels for a game it did not run (legacy/rulebook/tools/wm_tier0a_v2.py,
    pick_levels): the training level is the first with >= 500 recorded moves, the
    transfer level the next with >= 50 distinct cases. 1-based, as files are named."""
    import collections
    import glob
    counts = collections.Counter()
    for p in sorted(glob.glob(os.path.join(corpus, "shard_*.npz"))):
        with np.load(p) as z:
            for lv, c in zip(*np.unique(z["levels"], return_counts=True)):
                counts[int(lv)] += int(c)
    train = next((lv for lv in sorted(counts) if counts[lv] >= min_train_moves), min(counts))
    transfer = None
    for lv in sorted(x for x in counts if x > train):
        cand = LevelEvidence.from_corpus(game, corpus, lv)
        if cand is not None and len(cand.actions) >= min_transfer_cases:
            transfer = lv
            break
    return train + 1, None if transfer is None else transfer + 1


def conflicts(ev):
    return int((ev.n_variants > 1).sum())


def build(game, corpus, train_level, transfer_level, settings=DEFAULTS):
    """{split: LevelEvidence} for one game: "train" from the training level and
    "fit" / "score" from the transfer level (0-based levels, as read_moves
    counts them; transfer_level None: no transfer level). Each split is built
    from the recording with its scan mask passed through complete_bars, the way
    from_corpus and split build theirs. Also returns the report's numbers."""
    out, report = {}, {}

    def made(name, ev, scan):
        out[name] = ev
        plain = LevelEvidence.from_moves(game, ev.level, *raw[name], mask=scan, order=None)
        report[name] = {"mask_cells_scan": int(scan.sum()), "mask_cells": int(ev.mask.sum()),
                        "cases_scan": len(plain.actions), "cases": len(ev.actions),
                        "conflicting_scan": conflicts(plain), "conflicting": conflicts(ev)}
    raw = {}
    f, n, a, order = read_moves(corpus, train_level)
    scan, _ = ticker_mask(f[:MASK_MOVES], n[:MASK_MOVES])
    raw["train"] = (f, n, a)
    made("train", LevelEvidence.from_moves(game, train_level, f, n, a, mask=complete_bars(scan, settings),
                                           order=order), scan)
    if transfer_level is not None:
        fm = settings["fit_moves"]
        f, n, a, order = read_moves(corpus, transfer_level)
        part = lambda s: {k: v[s] for k, v in order.items()} if order else None     # attempts are the LEVEL's
        level_scan, _ = ticker_mask(f[:MASK_MOVES], n[:MASK_MOVES])
        fit_scan = ticker_mask(f[:fm], n[:fm])[0] | out["train"].mask       # what is known by then
        raw["fit"] = (f[:fm], n[:fm], a[:fm])
        made("fit", LevelEvidence.from_moves(game, transfer_level, *raw["fit"], mask=complete_bars(fit_scan, settings),
                                             order=part(slice(None, fm))), fit_scan)
        if len(a) > fm:
            raw["score"] = (f[fm:], n[fm:], a[fm:])
            score = LevelEvidence.from_moves(game, transfer_level, *raw["score"],
                                             mask=complete_bars(level_scan, settings), order=part(slice(fm, None)))
            score.first = np.array(f[0])              # the level's first board, not the split's (as split() does)
            score.first.flags.writeable = False
            score._api = None
            made("score", score, level_scan)
    return out, report


def run_game(run, game, corpus, train_level, transfer_level, log=print, force=False):
    """S0 for one game into `run`: the evidence files and the report. Levels are
    1-based here, as the evidence files are named (CP1's training level, its
    transfer level or None)."""
    settings = dict(DEFAULTS, **{k: v for k, v in run.config.get("mask", {}).items() if k in DEFAULTS})
    ev_dir = os.path.join(run.dir, "evidence")
    os.makedirs(ev_dir, exist_ok=True)
    stage = run.stage_dir("s0_mask")
    p_report = os.path.join(stage, f"{game}.json")
    shards = sorted(f for f in os.listdir(corpus) if f.startswith("shard_"))
    inputs_note = {"corpus": os.path.relpath(corpus, A.ROOT), "shards": shards,
                   "levels": [train_level, transfer_level]}
    key_settings = dict(settings, **inputs_note)
    if not force and os.path.exists(p_report) and run.is_current("s0_mask", game, VERSION, key_settings, {}):
        log(f"[{game}] S0 is current; skipped")
        return json.load(open(p_report))
    evs, report = build(game, corpus, train_level - 1, None if transfer_level is None else transfer_level - 1,
                        settings)
    paths = {}
    for split, ev in evs.items():
        k = train_level if split == "train" else transfer_level
        paths[split] = os.path.join(ev_dir, f"{game}_L{k}_{split}.npz")
        ev.save(paths[split])
    report = {"game": game, "settings": settings, **inputs_note, "splits": report}
    with open(p_report, "w") as f:
        json.dump(report, f, indent=1)
    run.write_header("s0_mask", game, VERSION, key_settings, {}, dict(paths, report=p_report), code=__file__)
    r = report["splits"]["train"]
    log(f"[{game}] S0: training level {train_level}: mask {r['mask_cells_scan']} -> {r['mask_cells']} cells, "
        f"conflicting cases {r['conflicting_scan']} -> {r['conflicting']}, cases {r['cases_scan']} -> {r['cases']}")
    return report
