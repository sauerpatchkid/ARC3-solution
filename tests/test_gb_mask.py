"""S0 MASK (custom_agents/gb/s0_mask.py): bars run to the screen's edges, nothing
else changes, and with the step switched off the evidence is CP1's exactly."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import cp1_import  # noqa: E402
from gb import s0_mask as M  # noqa: E402
from gb.core.evidence import LevelEvidence  # noqa: E402

ON, OFF = M.DEFAULTS, dict(M.DEFAULTS, complete_bars=False)


def empty():
    return np.zeros((64, 64), bool)


def test_a_bar_that_stops_short_is_run_to_the_edges():
    m = empty()
    m[63, :58] = True                               # m0r0's bar: the scan stopped at column 57
    out = M.complete_bars(m)
    assert out[63].all() and out.sum() == 64


def test_a_vertical_bar_and_two_bars_meeting_at_a_corner():
    m = empty()
    m[5:40, 0] = True
    assert M.complete_bars(m)[:, 0].all() and M.complete_bars(m).sum() == 64
    m[0, 0:30] = True                               # an L: each arm runs along its own edge, nothing more
    assert M.complete_bars(m).sum() == 64 + 64 - 1


def test_short_tickers_and_cells_away_from_the_edges_are_left_alone():
    m = empty()
    m[0, 10:15] = True                              # 5 cells: a ticker, not a bar
    m[30, 5:40] = True                              # a long run, but mid-screen
    assert np.array_equal(M.complete_bars(m), m)
    assert np.array_equal(M.complete_bars(m, OFF), m)


# ---- on CP1's recordings ------------------------------------------------------------------------------------
cp1 = pytest.mark.skipif(not os.path.isdir(cp1_import.CP1_DIR), reason="CP1's registered results are not here")


def levels(game):
    r = cp1_import.load_cp1(game)
    return r["corpus"], r["train"]["level"] - 1, r["transfer_level"] - 1


@cp1
@pytest.mark.parametrize("game", ["m0r0", "tu93"])
def test_switched_off_it_builds_cp1s_evidence_exactly(game):
    evs, _ = M.build(game, *levels(game), OFF)
    for split, ev in evs.items():
        theirs = np.load(os.path.join(cp1_import.CP1_DIR, f"{game}_{split}.npz"))
        mine = {"game": ev.game, "level": ev.level, "first": ev.first, "mask": ev.mask, "before": ev.before,
                "after": ev.after, "actions": ev.actions, "count": ev.count, "agree": ev.agree,
                "n_variants": ev.n_variants, "frozen": ev.frozen,
                **{"move_" + k: v for k, v in ev.moves.items()}}
        assert sorted(mine) == sorted(k for k in theirs.files if k != "info"), split
        for k, v in mine.items():
            assert np.array_equal(np.asarray(v), theirs[k]), (split, k)


@cp1
def test_switched_on_m0r0s_cases_no_longer_end_two_ways():
    _, report = M.build("m0r0", *levels("m0r0"), ON)
    train = report["train"]
    assert (train["mask_cells_scan"], train["mask_cells"]) == (122, 128)
    assert (train["conflicting_scan"], train["conflicting"]) == (13, 0)
