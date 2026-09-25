"""Unit tests for the upgrade screen's module (custom_agents/upgrades.py).

    uv run python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from canon import OnlineCanonicalizer  # noqa: E402
from upgrades import (Upgrades, UpgradedReturnMap, BarDetector, tick_cells,  # noqa: E402
                      label_components, screen_objects, DEAD_CLICK_TRIES)

NO_MASK = np.zeros((64, 64), dtype=bool)


# ---- decorations ------------------------------------------------------------------
def test_tick_cells_finds_small_components_only():
    d = np.zeros((64, 64), dtype=bool)
    d[5, 5] = True                    # size 1
    d[10, 10] = d[10, 11] = True      # size 2
    d[20, 20:23] = True               # size 3: not a tick
    d[30:36, 30:36] = True            # a big block: not a tick
    t = tick_cells(d)
    assert t[5, 5] and t[10, 10] and t[10, 11]
    assert not t[20, 20:23].any() and not t[30:36, 30:36].any()


def bar_game(steps, mid_mover=False):
    """A progress marker sweeping row 63 one cell per move (2 cells change, as
    on ft09), ALWAYS together with a 36-cell change in the playfield, so it
    never ticks alone and the shared detector cannot see it."""
    frames = []
    for t in range(steps + 1):
        f = np.zeros((64, 64), dtype=np.uint8)
        f[20:26, 20:26] = 4 if t % 2 else 5          # the real change every move
        f[63, t % 60] = 7                           # the bar
        if mid_mover:
            f[40, 10 + (t % 20) * 2] = 9            # a small mover mid-screen
        frames.append(f)
    return frames


def test_bar_detector_catches_a_bar_the_shared_detector_misses():
    fr = bar_game(1000)
    shared = OnlineCanonicalizer()
    bars = BarDetector()
    for a, b in zip(fr, fr[1:]):
        shared.update(a, b)
        bars.update(a, b)
    assert not shared.mask[63].any()                             # the known blind spot
    assert bars.sticky[63, :60].sum() >= 57
    assert not bars.sticky[:60].any()                            # nothing off the border band


def test_border_guard_rejects_mid_screen_ticks():
    fr = bar_game(1000, mid_mover=True)
    bars = BarDetector()
    for a, b in zip(fr, fr[1:]):
        bars.update(a, b)
    assert not bars.sticky[40].any()
    assert bars.sticky[63].any()


# ---- objects ------------------------------------------------------------------------
def test_label_components_small_case():
    g = np.array([[1, 1, 2], [2, 1, 2], [2, 2, 2]] + [[0, 0, 0]] * 61, dtype=np.uint8)
    g = np.pad(g, ((0, 0), (0, 61)))
    valid = np.zeros((64, 64), bool)
    valid[:3, :3] = True
    lab = label_components(g, valid)
    assert lab[0, 0] == lab[0, 1] == lab[1, 1] == 0
    assert lab[0, 2] == lab[1, 2] == lab[2, 2] == lab[1, 0] == lab[2, 0] == 2
    assert (lab[3:] == -1).all()


def test_screen_objects_orders_small_rare_objects_first_and_skips_background():
    f = np.zeros((64, 64), dtype=np.uint8)          # colour 0 is background
    f[10:30, 10:30] = 3                             # big, common
    f[50, 50] = 8                                   # tiny, rare
    labels, objs = screen_objects(f, NO_MASK)
    assert len(objs) == 2
    first_target = objs[0][1]
    assert divmod(first_target, 64) == (50, 50)
    y, x = divmod(objs[1][1], 64)
    assert 10 <= y < 30 and 10 <= x < 30            # target inside its object
    assert labels[0, 0] == -1                       # background is not an object


# ---- labels ---------------------------------------------------------------------------
def screen(i):
    f = np.zeros((64, 64), dtype=np.uint8)
    f[10, i] = 7
    return f


def drive(u, seq, reset_before=()):
    """Feed screens; return the label for each transition (None for the first)."""
    out, prev = [], None
    for t, i in enumerate(seq):
        if t in reset_before:
            u.on_game_over()
            prev = None
        cur = screen(i)
        u.observe(prev, cur, NO_MASK, 0 if prev is not None else None)
        out.append(u.label)
        prev = cur
    return out


def test_bars_only_label_is_binary_novelty():
    u = Upgrades(["bars"])
    assert drive(u, [0, 1, 0, 2]) == [None, 1.0, 0.0, 1.0]


def test_attempt_label_gives_half_credit_for_new_this_attempt():
    u = Upgrades(["attempt"])
    # attempt 1: 0 -> 1 -> 2 ; game over ; attempt 2: 0 -> 1 -> 3
    labs = drive(u, [0, 1, 2, 0, 1, 3], reset_before=(3,))
    assert labs == [None, 1.0, 1.0, None, 0.5, 1.0]


def test_graded_label_decays_with_visits():
    u = Upgrades(["graded"])
    labs = drive(u, [0, 1, 0, 1, 0])
    assert labs[1] == 1.0
    assert labs[2] == pytest.approx(1 / np.sqrt(2))
    assert labs[3] == pytest.approx(1 / np.sqrt(2))
    assert labs[4] == pytest.approx(1 / np.sqrt(3))


def test_reward_passes_through_without_a_label_option():
    u = Upgrades(["deadclick"])
    drive(u, [0, 1])
    assert u.reward(1.0) == 1.0 and u.reward(0.0) == 0.0


def test_level_change_resets_label_memory():
    u = Upgrades(["bars"])
    drive(u, [0, 1])
    u.on_level()
    assert drive(u, [0, 1]) == [None, 1.0]


# ---- dead clicks -------------------------------------------------------------------------
def test_dead_click_is_blocked_after_four_no_ops_and_resampled():
    u = Upgrades(["deadclick"], rng=np.random.RandomState(0))
    f = screen(0)
    click = 5 + 64 * 10 + 20                        # (10, 20), colour 0 there
    u.observe(None, f, NO_MASK, None)
    for _ in range(DEAD_CLICK_TRIES):
        u.observe(f, f, NO_MASK, click)              # the click changed nothing
    assert (10, 20, 0) in u.blocked
    probs = np.zeros(5 + 4096, dtype=np.float32)
    probs[click] = 1.0
    probs[2] = 1.0                                   # ACTION3 also possible
    a = u.override(5, (10, 20), 64 * 10 + 20, probs, f)
    assert a == (2, None, None)                      # never the dead click
    g = f.copy(); g[10, 20] = 3                      # same cell, new colour: allowed
    assert u.override(5, (10, 20), 64 * 10 + 20, probs, g) == (5, (10, 20), 64 * 10 + 20)
    u.on_level()
    assert not u.blocked


def test_map_options_need_the_map():
    u = Upgrades(["map_gated"])
    with pytest.raises(SystemExit):
        u.check(map_on=False)
    with pytest.raises(SystemExit):
        Upgrades(["nonsense"])


# ---- the upgraded map -----------------------------------------------------------------------
def walk(m, seq, avail=(1, 2, 3, 4), action=0):
    has_prev = False
    for i in seq:
        m.observe(screen(i), NO_MASK, has_prev, list(avail))
        m.record(action)
        has_prev = True


def test_gated_map_alternates_then_follows_the_better_arm():
    m = UpgradedReturnMap(stall=10**9, gated=True)
    # "walk" attempts find many new screens; "stay" attempts find none
    for attempt in range(40):
        arm = m.gate_arm
        m.attempt_steps = 10
        m.attempt_new = 5 if arm == "walk" else 0
        m.last_find = None
        m.on_game_over()
    walks, stays = m.stats["gate_walk"], m.stats["gate_stay"]
    assert walks > stays >= 3


def test_gated_map_stops_walking_back_when_it_does_not_pay():
    m = UpgradedReturnMap(stall=10**9, gated=True)
    for attempt in range(40):
        m.attempt_steps = 10
        m.attempt_new = 0 if m.gate_arm == "walk" else 5
        m.on_game_over()
    assert m.stats["gate_stay"] > m.stats["gate_walk"]
    if m.gate_arm == "stay":
        assert not m.return_pending


def test_object_map_counts_untried_objects_and_clicks_one_on_arrival():
    m = UpgradedReturnMap(stall=10**9, objects=True)
    f = np.zeros((64, 64), dtype=np.uint8)
    f[5, 5] = 8
    f[40, 40] = 9
    k = m.observe(f, NO_MASK, False, [6])
    assert len(m.obj_targets[k]) == 2 and m._is_frontier(k)
    m.record(5 + 64 * 5 + 5)                          # click the first object
    assert m._is_frontier(k)
    m.arrived = True
    a = m.choose()
    assert a == 5 + 64 * 40 + 40                      # the object not yet clicked
    m.record(a)
    assert not m._is_frontier(k)


def test_diverse_map_picks_a_reachable_frontier():
    m = UpgradedReturnMap(stall=10**9, diverse=True, rng=np.random.RandomState(0))
    m.attempts.extend([100] * 3)
    walk(m, [0, 1, 2, 3])
    k0 = m.key(screen(0), NO_MASK)
    path = m._return_route(k0)
    assert path and 1 <= len(path) <= 3


def test_bar_mask_reaches_the_map_fingerprint():
    bars = BarDetector()
    bars.sticky[63, :] = True
    m = UpgradedReturnMap(stall=10**9, bars=bars)
    a, b = screen(0), screen(0)
    a[63, 1] = 7
    b[63, 9] = 7
    assert m.key(a, NO_MASK) == m.key(b, NO_MASK)
