"""tickers.py — component-level decorative-cell detection.

WHY THIS EXISTS (and does not just call metrics_common.find_indicator_cells).

The shared canonicalizer has two branches:

  A) FIXED   — a cell that changes in >=95% of ALL transitions.
  B) ROTATING— a compact cell set covering >=95% of the changes seen in
               "tiny" transitions, where tiny means the WHOLE TRANSITION
               changed <=2 cells, and only if tiny transitions are >=30% of
               the run.

Branch B's unit is the transition. That is the bug: a ticker only becomes
visible to it when the ticker ticks *alone*. Measured on ft09 (the flagship
game), row 63 changes in 85.4% of transitions, always exactly 2 cells, with a
near-uniform per-cell count across the full width — a progress bar sweeping the
screen. It is never alone: it co-occurs with the 36-cell tile toggle the agent's
click causes, so every such transition has 38 changed cells, `tiny_frac` is
0.000, branch B never fires, and branch A cannot fire either because each bar
cell is only visited once per sweep (max per-cell frequency 0.064). ft09 is
scored with an EMPTY decorative mask. So are ls20, ar25, lp85 and dc22.

This module changes the unit from the transition to the CONNECTED COMPONENT: a
small component inside a large transition is still a tick. Branch A is kept
verbatim; branch B is generalised. On ft09 this recovers row 63.

Consequence for existing numbers: masking row 63 cuts ft09's unique canonical
states from 81,916 to 59,999 over a 96,411-action run (1.4x inflation in
`unique_states_per_action`, the headline coverage metric). This module does NOT
change metrics_common — backporting it would move every published baseline
number and is a decision for the advisor, not a side effect of the LLM track.
Here it feeds C1's "is this change decorative" feature, which is otherwise dead
on exactly the games the track cares about.
"""
import numpy as np

from metrics_common import DECOR_THRESHOLD, GRID, N_CELLS

# A changed-cell component of at most this many cells counts as a "tick"
# candidate. Deliberately identical to metrics_common's <=2, so this module's
# ONLY difference from the shared canonicalizer is the unit (component, not
# whole transition) and nothing else has to be re-argued. Measured at 4 it
# false-positives badly: lp85's board is 2x2 tiles on a regular lattice, and a
# 4-cell threshold masks 162 cells of live playfield (rows 17-45) as decoration.
MAX_TICK_CELLS = 2
# Fraction of transitions that must contain a tick candidate before the
# rotating branch is trusted at all. Mirrors metrics_common's 0.30 guard.
MIN_TICK_FRAC = 0.30
# Fraction of tick-candidate cell-changes the mask must cover.
TICK_COVERAGE = 0.95
# Hard cap on the rotating mask, so a game whose whole board flickers in small
# pieces cannot mask itself into nothing.
MAX_MASK_CELLS = 256

# Only half the 8-neighbourhood is needed: each pair is visited once.
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1))


def connected_components(mask):
    """8-connected components of a bool mask, as a list of (n, 2) int16 arrays.

    Union-find over only the True cells. Changed-cell counts on the Stage-1
    games run 0-577, so this is microseconds per transition — cheap enough for
    the agent's per-step path (budget is ~5 ms of model time per action).
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    cells = list(zip(ys.tolist(), xs.tolist()))
    index = {c: i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (y, x), i in index.items():
        for dy, dx in _NEIGHBOURS:
            j = index.get((y + dy, x + dx))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups = {}
    for c, i in index.items():
        groups.setdefault(find(i), []).append(c)
    return [np.array(v, dtype=np.int16) for v in groups.values()]


class TickerScan:
    """Streaming accumulator: transitions in, decorative-cell mask out.

    Usage:
        scan = TickerScan()
        for s in iter_shards(d, fields=("frames", "next_frames")):
            scan.update(s["frames"], s["next_frames"])
        mask, info = scan.result()
    """

    def __init__(self, max_tick_cells=MAX_TICK_CELLS, min_tick_frac=MIN_TICK_FRAC,
                 coverage=TICK_COVERAGE, max_mask_cells=MAX_MASK_CELLS):
        self.max_tick_cells = max_tick_cells
        self.min_tick_frac = min_tick_frac
        self.coverage = coverage
        self.max_mask_cells = max_mask_cells
        self.n = 0
        self.n_tick = 0                                        # transitions with >=1 tick candidate
        self.diff_count = np.zeros((GRID, GRID), dtype=np.int64)
        self.tick_count = np.zeros((GRID, GRID), dtype=np.int64)

    def update(self, frames, next_frames):
        diff = frames != next_frames
        self.n += diff.shape[0]
        self.diff_count += diff.sum(axis=0).astype(np.int64)
        for d in diff:
            if not d.any():
                continue
            hit = False
            for comp in connected_components(d):
                if len(comp) <= self.max_tick_cells:
                    self.tick_count[comp[:, 0], comp[:, 1]] += 1
                    hit = True
            if hit:
                self.n_tick += 1

    def result(self):
        """Return (mask, info) — mask is (64, 64) bool, info a stats dict."""
        if self.n == 0:
            return np.zeros((GRID, GRID), dtype=bool), {"n": 0}
        freq = self.diff_count / self.n
        fixed = freq >= DECOR_THRESHOLD
        tick_frac = self.n_tick / self.n
        rot = np.zeros(N_CELLS, dtype=bool)
        k = 0
        if tick_frac >= self.min_tick_frac:
            counts = self.tick_count.ravel().astype(float)
            total = counts.sum()
            if total > 0:
                order = np.argsort(-counts)
                covered = np.cumsum(counts[order]) / total
                k = int(np.searchsorted(covered, self.coverage)) + 1
                if k <= self.max_mask_cells:
                    rot[order[:k]] = True
                else:
                    k = 0                                      # too diffuse to be decoration
        mask = fixed | rot.reshape(GRID, GRID)
        return mask, {
            "n": self.n,
            "tick_frac": round(tick_frac, 4),
            "fixed_cells": int(fixed.sum()),
            "rotating_cells": int(rot.sum()),
            "rotating_candidate_k": k,
            "mask_cells": int(mask.sum()),
            "max_cell_freq": round(float(freq.max()), 4),
        }


def scan_corpus(corpus_dir, limit=None, **kw):
    """Convenience: run a TickerScan over a corpus directory.

    CAUTION: the mask depends on how much of the run is scanned, because a
    ticker may only tick alone during some phases. Measured on lp85, 25 shards
    give a 60-cell mask (a vertical bar in column 0, tick_frac 0.39) and 50
    shards give an EMPTY mask (tick_frac 0.20, under the 0.30 guard). Compute
    the mask once per corpus over a fixed, recorded window and store it with the
    labels; never recompute it ad hoc, or g is trained against one mask and
    applied under another. See llm_track/SCHEMA.md.
    """
    from .corpus import iter_shards
    scan = TickerScan(**kw)
    for s in iter_shards(corpus_dir, fields=("frames", "next_frames"), limit=limit):
        scan.update(s["frames"], s["next_frames"])
    return scan.result()
