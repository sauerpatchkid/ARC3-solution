"""canon.py — online canonical-state identity for StochasticGoose (Plan B, §4.1).

The offline scorer (compute_metrics.py) decides which cells are decorative
"indicator" cells (tickers, progress bars) from whole-run aggregates and masks
them before counting unique states. This module runs the SAME detector,
metrics_common.find_indicator_cells, on RUNNING aggregates, so an agent can
ask "have I seen this screen before?" while it plays, and the scorer's notion
of a state and the agent's agree.

    canon = OnlineCanonicalizer()
    ...
    canon.update(prev_raw, cur_raw)   # once per transition, raw uint8 (64, 64)
    key = canon.key(cur_raw)          # int; equal keys <=> same canonical state

Before `warmup` transitions the mask is all-False, so key == hash of the raw
frame. Every `refresh` transitions after that the mask is recomputed from the
running counts, so a cell that stops changing drops out and one that starts
ticking is picked up. Keys made under different masks are not comparable in
general; callers keep their `seen` sets across a refresh anyway (the plan
accepts that lag), and the diagnostic in tools/label_diagnostic.py measures
how far the end-of-run online mask is from the offline one.

Pure numpy + xxhash; nothing here touches torch or the agent. NOT imported by
custom_agents/action.py yet — that wiring is Plan B step 2 and sits behind
EVAL_LABEL / EVAL_MASK_TRIED, which default to the baseline.
"""
import numpy as np
import xxhash

from metrics_common import find_indicator_cells, GRID

DEFAULT_WARMUP = 200    # EVAL_CANON_WARMUP: transitions before any masking
DEFAULT_REFRESH = 250   # EVAL_CANON_REFRESH: recompute cadence after warm-up


class OnlineCanonicalizer:
    def __init__(self, warmup=DEFAULT_WARMUP, refresh=DEFAULT_REFRESH):
        self.warmup = int(warmup)
        self.refresh = int(refresh)
        self.n_trans = 0
        self.cell_change_count = np.zeros((GRID, GRID), dtype=np.int64)
        self.tiny_count = 0
        self.tiny_cell_counts = np.zeros((GRID, GRID), dtype=np.int64)
        self.mask = np.zeros((GRID, GRID), dtype=bool)
        self.n_refreshes = 0

    # -- aggregates -------------------------------------------------------
    def update(self, prev_raw, cur_raw):
        """Record one transition. Refreshes the mask when due."""
        diff = prev_raw != cur_raw
        self._absorb(diff[None], 1)

    def update_batch(self, prev_raw, cur_raw):
        """Record a batch of transitions at once (offline replay only)."""
        diff = prev_raw != cur_raw
        self._absorb(diff, diff.shape[0])

    def _absorb(self, diff, m):
        self.cell_change_count += diff.sum(axis=0).astype(np.int64)
        per_t = diff.sum(axis=(1, 2))
        tiny = (per_t > 0) & (per_t <= 2)
        k = int(tiny.sum())
        if k:
            self.tiny_count += k
            self.tiny_cell_counts += diff[tiny].sum(axis=0).astype(np.int64)
        before = self.n_trans
        self.n_trans += m
        # Refresh once per crossing of a refresh boundary past warm-up.
        if self.n_trans >= self.warmup and (
            before < self.warmup
            or (self.n_trans // self.refresh) > (before // self.refresh)
        ):
            self.recompute_mask()

    def recompute_mask(self):
        if self.n_trans == 0:
            return
        freq = self.cell_change_count / self.n_trans
        self.mask = find_indicator_cells(
            freq=freq, tiny_frac=self.tiny_count / self.n_trans,
            tiny_cell_counts=self.tiny_cell_counts)
        self.n_refreshes += 1

    # -- identity ---------------------------------------------------------
    def canonical(self, raw):
        g = np.array(raw, dtype=np.uint8, copy=True)
        g[self.mask] = 0
        return g

    def key(self, raw):
        return xxhash.xxh64(self.canonical(raw).tobytes()).intdigest()

    def keys_batch(self, raws):
        """Keys for a (m, 64, 64) stack under the CURRENT mask."""
        g = np.array(raws, dtype=np.uint8, copy=True)
        g[:, self.mask] = 0
        return [xxhash.xxh64(row.tobytes()).intdigest() for row in g]

    def stats(self):
        return {"n_trans": self.n_trans, "masked_cells": int(self.mask.sum()),
                "n_refreshes": self.n_refreshes}
