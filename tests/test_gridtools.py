"""Unit tests for custom_agents/gridtools.py (shared 64x64 screen helpers).
The tick_cells / label_components / screen_objects cases written with the
upgrade screen are in tests/test_upgrades.py."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gridtools import GRID, border_band, connected_components, tick_cells  # noqa: E402


def test_border_band_is_the_outer_ring():
    b = border_band(4)
    assert b[0, 30] and b[3, 30] and b[60, 30] and b[30, 0] and b[30, 63]
    assert not b[4, 30] and not b[30, 4] and not b[59, 59]
    assert int(b.sum()) == GRID * GRID - (GRID - 8) ** 2


def test_connected_components_joins_diagonals():
    m = np.zeros((GRID, GRID), bool)
    m[2, 2] = m[3, 3] = True                 # diagonal neighbours: one blob
    m[10, 10] = m[10, 11] = m[10, 12] = True
    m[40, 5] = True
    sizes = sorted(len(c) for c in connected_components(m))
    assert sizes == [1, 2, 3] and connected_components(np.zeros((GRID, GRID), bool)) == []


def test_tick_cells_equals_the_blobs_of_at_most_two_cells():
    """The vectorised tick scan must agree with the plain definition (every
    8-connected changed blob of <= 2 cells) - the Rulebook's evidence index
    switched from the second to the first on 2026-10-06."""
    rng = np.random.RandomState(0)
    for density in (0.002, 0.01, 0.05, 0.2):
        for _ in range(25):
            d = rng.rand(GRID, GRID) < density
            want = np.zeros_like(d)
            for comp in connected_components(d):
                if len(comp) <= 2:
                    want[comp[:, 0], comp[:, 1]] = True
            assert np.array_equal(tick_cells(d), want)
