"""Unit tests for Option 1's ReturnMap (custom_agents/return_map.py).

    uv run python -m pytest tests/ -q

Screens are synthetic 64x64 frames; a "screen id" is painted into one cell so
distinct ids give distinct keys. The live decoration mask is all-False unless a
test says otherwise.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from return_map import ReturnMap, DEAD  # noqa: E402

NO_MASK = np.zeros((64, 64), dtype=bool)
BUTTONS = [1, 2, 3, 4]          # ACTION1-4 available, no clicks (values, as the frame lists them)


def screen(i, bar=None):
    f = np.zeros((64, 64), dtype=np.uint8)
    f[10 + i // 60, i % 60] = 7          # the "position" of the screen
    if bar is not None:
        f[63, bar % 60] = 3              # a progress-bar cell
    return f


class Walker:
    """Drives a ReturnMap the way the agent does: observe -> choose -> record."""

    def __init__(self, m, avail=BUTTONS):
        self.m, self.avail, self.has_prev = m, avail, False

    def see(self, i, mask=NO_MASK, bar=None):
        k = self.m.observe(screen(i, bar), mask, self.has_prev, self.avail)
        return k

    def act(self, goose_choice):
        a = self.m.choose()
        a = goose_choice if a is None else a
        self.m.record(a)
        self.has_prev = True
        return a


def corridor(m, n=5):
    """Walk screens 0, 1, ..., n pressing button 0 (ACTION1) each time."""
    w = Walker(m)
    for i in range(n):
        w.see(i)
        w.act(0)
    w.see(n)
    return w


def test_edges_and_shortest_route():
    m = ReturnMap(stall=10**9)
    w = corridor(m, 4)
    k = [m.key(screen(i), NO_MASK) for i in range(5)]
    assert m.edges[k[0]][0] == k[1] and m.edges[k[3]][0] == k[4]
    # shortest route from screen 0 to screen 3 is three presses of button 0
    path = m._bfs(k[0], lambda s: s == k[3])
    assert [a for a, _ in path] == [0, 0, 0]
    assert [n for _, n in path] == k[1:4]
    assert m._bfs(k[4], lambda s: s == k[0]) is None      # no known way back


def test_stall_routes_to_nearest_untried_button_and_tries_it():
    m = ReturnMap(stall=3, rng=np.random.RandomState(0))
    w = Walker(m)
    k0 = w.see(0); w.act(0)
    k1 = w.see(1); w.act(0)
    k2 = w.see(2)
    # screen 0 and screen 2: every button tried. Screen 1: button 3 never pressed.
    m.edges[k0].update({1: k0, 2: k0, 3: k0})
    m.edges[k1].update({1: k0, 2: k1})
    m.edges[k2].update({0: k2, 1: k1, 2: k2, 3: k2})
    m.since_new = 3                                   # stalled on screen 2
    assert w.act(goose_choice=99) == 1                # route: 2 -button 1-> 1
    assert m.stats["stall_routes"] == 1
    w.see(1)                                          # the game agrees
    assert w.act(goose_choice=99) == 3                # arrived: the untried button
    assert m.stats["routes_completed"] == 1 and m.stats["routes_aborted"] == 0


def test_route_arrival_tries_untried_button():
    m = ReturnMap(stall=1, rng=np.random.RandomState(0))
    w = Walker(m)
    k0 = w.see(0); w.act(0)
    k1 = w.see(1); w.act(0)
    k2 = w.see(2)
    # screen 0: all buttons tried (loop back to itself) except none -> not frontier
    for a in (1, 2, 3):
        m.edges[k0][a] = k0
    # screen 1: only button 0 tried -> frontier; screen 2: everything tried
    for a in range(4):
        m.edges[k2].setdefault(a, k2)
    m.edges[k2][1] = k0
    m.since_new = 1
    first = w.act(goose_choice=99)   # route: 2 -a1-> 0 -a0-> 1
    assert first == 1
    w.see(0)
    second = w.act(goose_choice=99)
    assert second == 0
    w.see(1)                         # arrived at the frontier
    third = w.act(goose_choice=99)
    assert third in (1, 2, 3)        # an untried button at screen 1
    assert m.stats["routes_completed"] == 1 and m.stats["frontier_button_tries"] == 1


def test_route_aborts_when_the_game_disagrees():
    m = ReturnMap(stall=10**9)
    w = corridor(m, 3)
    k = [m.key(screen(i), NO_MASK) for i in range(4)]
    m.route = [(0, k[1]), (0, k[2])]
    m.cur = k[0]
    a = m.choose(); m.record(a)
    assert m.expect == k[1]
    m.observe(screen(9), NO_MASK, True, BUTTONS)     # landed somewhere else
    assert m.route == [] and m.stats["routes_aborted"] == 1
    assert m.choose() is None                        # Goose plays normally


def test_unavailable_route_step_aborts():
    m = ReturnMap(stall=10**9)
    corridor(m, 3)
    k = [m.key(screen(i), NO_MASK) for i in range(4)]
    m.observe(screen(0), NO_MASK, False, [2, 3])     # button 0 (ACTION1) not available now
    m.route = [(0, k[1])]
    assert m.choose() is None and m.stats["routes_aborted"] == 1


def test_game_over_marks_the_move_and_returns_to_latest_discovery():
    m = ReturnMap(stall=10**9)
    m.attempts.extend([100, 100, 100])       # attempts usually last ~100 moves
    w = corridor(m, 3)                       # discovered 1, 2, 3 in that order; on 3
    k = [m.key(screen(i), NO_MASK) for i in range(4)]
    w.act(2)                                 # button 2 from screen 3, then game over
    m.on_game_over()
    assert m.edges[k[3]][2] == DEAD and m.return_pending
    assert m.return_limit() == 50
    w.has_prev = False                       # the level restarts at screen 0
    w.see(0)
    steps = [w.act(goose_choice=99)]
    for i in (1, 2):
        w.see(i)
        steps.append(w.act(goose_choice=99))
    assert steps == [0, 0, 0] and m.stats["return_routes"] == 1
    w.see(3)
    nxt = w.act(goose_choice=99)
    assert nxt in (0, 1, 3)                  # an untried button at screen 3, never the DEAD one
    assert m.stats["routes_completed"] == 1


def test_walk_back_leaves_half_the_attempt_for_exploring():
    m = ReturnMap(stall=10**9)
    w = corridor(m, 3)                       # a 4-move attempt: 0, 1, 2, 3
    k = [m.key(screen(i), NO_MASK) for i in range(4)]
    w.act(2)
    m.on_game_over()
    assert list(m.attempts) == [4] and m.return_limit() == 2
    w.has_prev = False
    w.see(0)
    assert w.act(goose_choice=99) == 0       # heads for screen 2, not screen 3
    w.see(1)
    assert w.act(goose_choice=99) == 0
    w.see(2)
    assert w.act(goose_choice=99) in (1, 2, 3)   # arrived at screen 2: untried button
    assert m.stats["route_len_total"] == 2


def test_no_walk_back_when_switched_off():
    m = ReturnMap(stall=10**9, return_after_game_over=False)
    w = corridor(m, 3)
    w.act(2)
    m.on_game_over()
    w.has_prev = False
    w.see(0)
    assert w.act(goose_choice=99) == 99 and m.stats["return_routes"] == 0


def test_dead_moves_are_never_routed_through():
    m = ReturnMap(stall=10**9)
    corridor(m, 3)
    k = [m.key(screen(i), NO_MASK) for i in range(4)]
    m.edges[k[1]][0] = DEAD
    assert m._bfs(k[0], lambda s: s == k[3]) is None


def test_click_screen_is_frontier_until_click_tries():
    m = ReturnMap(stall=10**9, click_tries=3)
    k = m.observe(screen(0), NO_MASK, False, [6])    # click-only screen
    assert m._is_frontier(k)
    for _ in range(3):
        m.record(5 + 64 * 10 + 10)
    assert not m._is_frontier(k)


def test_level_change_forgets_everything():
    m = ReturnMap(stall=10**9)
    corridor(m, 3)
    m.clear()
    assert m.edges == {} and m.last is None and not m.return_pending
    assert m.stats["levels"] == 1


def test_sticky_mask_ignores_progress_bar_and_resets_when_it_grows():
    m = ReturnMap(stall=10**9)
    bar_mask = np.zeros((64, 64), dtype=bool)
    bar_mask[63, :60] = True
    # before the bar is known, the bar changes the key
    assert m.key(screen(0, bar=1), NO_MASK) != m.key(screen(0, bar=2), NO_MASK)
    corridor(m, 2)
    assert m.edges
    # the live mask learns the bar: the map starts over, and the bar no longer matters
    k1 = m.key(screen(0, bar=1), bar_mask)
    assert m.edges == {} and m.stats["mask_resets"] == 1
    assert k1 == m.key(screen(0, bar=5), bar_mask)
    # the live mask flipping OFF a cell does not shrink the sticky mask or reset again
    flipped = bar_mask.copy(); flipped[63, 7] = False
    assert m.key(screen(0, bar=7), flipped) == k1
    assert m.stats["mask_resets"] == 1


def test_unpack_matches_the_sampler_format():
    assert ReturnMap.unpack(3) == (3, None, None)
    assert ReturnMap.unpack(5 + 64 * 12 + 40) == (5, (12, 40), 64 * 12 + 40)


def test_override_keeps_goose_choice_when_map_is_idle():
    m = ReturnMap(stall=10**9)
    m.observe(screen(0), NO_MASK, False, BUTTONS)
    assert m.override(2, None, None) == (2, None, None)
    assert m.stats["map_actions"] == 0
