"""metrics_common.py — shared indicator-cell detection for both scoring pipelines.

The canonicalizer detects two kinds of decorative cell:
  A) FIXED ticker: a cell changing in >=DECOR_THRESHOLD of all transitions.
  B) ROTATING ticker: a compact cell set that accounts for nearly all
     "tiny" transitions (<=2 cells changed). See find_indicator_cells().

Used by compute_metrics.py (streaming) and summarize_runs.py (batch).
"""
import numpy as np

GRID = 64
N_CELLS = GRID * GRID
DECOR_THRESHOLD = 0.95


def find_indicator_cells(*, freq, tiny_frac, tiny_cell_counts):
    """Detect indicator (decorative) cells from pre-computed aggregates.

    Parameters
    ----------
    freq : ndarray (64, 64)
        Per-cell fraction of transitions in which the cell changed.
    tiny_frac : float
        Fraction of transitions that changed >0 and <=2 cells.
    tiny_cell_counts : ndarray (64, 64)
        Per-cell count of changes summed over only the "tiny" transitions.

    Returns
    -------
    mask : ndarray (64, 64) of bool
        True for cells identified as decorative indicators.
    """
    fixed = freq >= DECOR_THRESHOLD
    if tiny_frac >= 0.30:
        cellcounts = tiny_cell_counts.ravel().astype(float)
        total = cellcounts.sum()
        if total > 0:
            order = np.argsort(-cellcounts)
            covered = np.cumsum(cellcounts[order]) / total
            k = int(np.searchsorted(covered, 0.95)) + 1
            if k <= 128:
                rot = np.zeros(N_CELLS, dtype=bool)
                rot[order[:k]] = True
                return fixed | rot.reshape(GRID, GRID)
    return fixed
