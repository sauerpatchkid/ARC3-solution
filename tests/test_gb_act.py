"""S5 ACT (custom_agents/gb/s5_act.py): carried no-ops, goal hints, the reweighting,
the rules' child process, the map's part, the online bookkeeping and the replay.
Synthetic boards only; no GPU, no LLM, no recorded game.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import s5_act as S  # noqa: E402
from return_map import ReturnMap  # noqa: E402

NO_MASK = np.zeros((64, 64), bool)

# A v1 rule: a click on colour 4 changes nothing (it claims the clicked cell, unchanged).
CLICK4_NOOP = '''
RULE = "clicking colour 4 does nothing"
def applies(board, act, api):
    return act.action == 6 and board[act.click] == 4
def predict(board, act, api):
    claimed = np.zeros((64, 64), bool)
    claimed[act.click] = True
    return board.copy(), claimed
'''
HANGS = CLICK4_NOOP.replace("    return act.action == 6 and board[act.click] == 4",
                            "    while True:\n        pass")


def board(fill=0, cells=()):
    b = np.full((64, 64), fill, np.uint8)
    for (y, x, c) in cells:
        b[y, x] = c
    return b


def click(y, x):
    return 5 + 64 * y + x


# ---- colour no-ops and goal hints --------------------------------------------------------------
def test_colour_noop_is_carried_after_enough_clicks_and_dropped_at_a_contradiction():
    g = S.Guide("toy", settings={"noop_min_clicks": 20})
    b = board(0, [(5, 5, 3)])
    g.new_level(1, b)
    for _ in range(19):
        g.observe(b, click(5, 5), b, changed=False)
    g.new_level(2, b)
    assert g.covered(b, click(5, 5)) == ()                 # 19 clicks: not enough
    for _ in range(20):
        g.observe(b, click(5, 5), b, changed=False)
    g.new_level(3, b)
    assert g.covered(b, click(5, 5)) == ("colour 3",)
    assert g.covered(b, 0) == ()                            # a button is never a colour no-op
    after = b.copy()
    after[0, 0] = 9
    g.observe(b, click(5, 5), after, changed=True)          # it did something: dropped
    assert g.covered(b, click(5, 5)) == ()
    assert [e["kind"] for e in g.events if "colour" in e["kind"]] == ["colour no-op carried", "colour no-op dropped"]


def test_a_colour_that_ever_changed_something_is_not_carried():
    g = S.Guide("toy")
    b = board(0, [(5, 5, 3)])
    g.new_level(1, b)
    for i in range(30):
        g.observe(b, click(5, 5), b, changed=(i == 7))
    g.new_level(2, b)
    assert g.covered(b, click(5, 5)) == ()


def test_goal_colours_are_the_ones_that_vanished_or_shrank_most():
    first = board(0, [(1, 1, 5), (1, 2, 5), (2, 2, 7), (3, 3, 7), (3, 4, 7), (9, 9, 8)])
    before_win = board(0, [(9, 9, 8), (3, 3, 7)])           # 5 vanished (2 cells), 7 shrank by 2, 8 kept
    assert S.goal_colours(first, before_win, NO_MASK) == [5, 7]
    assert S.goal_colours(first, first, NO_MASK) == []
    assert S.goal_colours(board(3), board(0, [(0, 0, 3)]), NO_MASK) == []    # the background is never a hint


def test_a_hint_boosts_nothing_on_a_screen_with_no_hint_coloured_click():
    g = S.Guide("toy", seed=0)
    b = board(0, [(1, 1, 5)])
    g.new_level(1, b, NO_MASK)
    g.hints = {5: 0}
    probs = np.zeros(4101)
    probs[0] = probs[1] = 1.0                                # buttons only: no click is possible
    for _ in range(200):
        g.override(0, None, None, probs, b)
    assert g.stats["changed_move"] == 0


def test_a_hint_boosts_clicks_on_its_colour_and_is_dropped_after_quiet_moves():
    g = S.Guide("toy", settings={"hint_drop_moves": 5, "gamma": 1.0})
    first = board(0, [(1, 1, 5)])
    g.new_level(1, first, NO_MASK)
    g.new_level(2, first, NO_MASK, before_win=board(0))
    assert g.hints == {5: 0}
    assert g.weight(first, click(1, 1)) == 2.0 and g.weight(first, click(2, 2)) == 1.0
    for _ in range(5):
        g.observe(first, 0, first, changed=False)
    assert g.hints == {}


# ---- the reweighting ---------------------------------------------------------------------------
def test_override_draws_with_probability_p_times_w():
    """Two moves with Goose's probability 1/2 each; one is covered (alpha = 0.1), so it
    should be taken with probability 0.05 / 0.55."""
    g = S.Guide("toy", settings={"alpha": 0.1}, seed=1)
    b = board(0, [(0, 1, 3)])
    g.colour_noops = {3: 1}
    g.new_level(2, b)
    g.colour_noops = {3: 1}
    probs = np.zeros(5 + 4096)
    probs[5 + 0], probs[5 + 1] = 1.0, 1.0                   # clicks on (0, 0) colour 0 and (0, 1) colour 3
    goose = np.random.default_rng(0)
    p = probs.copy()
    p[5:] /= 4096
    p /= p.sum()
    taken = []
    for _ in range(20000):
        a = int(goose.choice(len(p), p=p))
        idx, coords, cidx = ReturnMap.unpack(a)
        got = g.override(idx, coords, cidx, probs, b)
        taken.append(got[2])
    share = np.mean(np.array(taken) == 1)
    assert share == pytest.approx(0.05 / 0.55, abs=0.01)


def test_override_leaves_an_uncovered_move_alone_and_survives_an_empty_distribution():
    g = S.Guide("toy")
    b = board(0)
    g.new_level(1, b)
    assert g.override(2, None, None, np.zeros(4101), b) == (2, None, None)
    g.colour_noops = {0: 1}                                 # everything covered, but nothing to draw from
    assert g.override(5, (0, 0), 0, np.zeros(4101), b) == (5, (0, 0), 0)


# ---- rules in the child process ------------------------------------------------------------------
def test_the_rule_worker_answers_and_a_hanging_rule_switches_the_rules_off():
    w = S.RuleWorker(timeout_s=3.0)
    try:
        b = board(0, [(2, 3, 4)])
        assert w.level(b, NO_MASK, 0, [(7, CLICK4_NOOP, {})]) == 1
        assert w.ask(b, click(2, 3), {7}) == [7]
        assert w.ask(b, click(0, 0), {7}) == []
        assert w.ask(b, 1, {7}) == []
    finally:
        w.stop()
    w = S.RuleWorker(timeout_s=1.0)
    try:
        assert w.level(b, NO_MASK, 0, [(8, HANGS, {})]) == 1
        assert w.ask(b, click(2, 3), {8}) == [] and w.dead
        assert w.ask(b, click(2, 3), {8}) == []             # stays off
    finally:
        w.stop()


def test_a_noop_rule_waits_is_rebound_on_the_level_and_covers_moves():
    rule = {"id": 3, "group": "click on colour 4 (dark gray)", "code": CLICK4_NOOP, "binding": {}}
    g = S.Guide("toy", [rule], train_level=1, settings={"online_min_cases": 2, "online_max_wait": 10_000,
                                                         "noop_rule_min_cases": 3})
    try:
        b = board(0, [(2, y, 4) for y in range(8)] + [(9, 9, 6)])
        g.new_level(1, b, NO_MASK)
        assert g.pending == {3} and g.covered(b, click(2, 0)) == ()
        for y in range(8):                                  # clicks on colour 4: nothing changes
            g.observe(b, click(2, y), b, changed=False)
        moved = b.copy()
        moved[9, 9], moved[9, 10] = 0, 6
        for x in range(2):                                  # two distinct changing clicks: the wait ends
            g.observe(b if x == 0 else moved, click(9, 9 + x), moved if x == 0 else b, changed=True)
        assert g.active == {3: {}}, g.events
        assert g.covered(b, click(2, 5)) == ("rule 3",)
        assert g.covered(b, click(9, 9)) == ()
    finally:
        g.close()


def test_a_rule_with_too_few_cases_is_tried_again_with_more_of_the_level():
    rule = {"id": 3, "group": "click on colour 4 (dark gray)", "code": CLICK4_NOOP, "binding": {}}
    g = S.Guide("toy", [rule], train_level=1, settings={"online_min_cases": 1, "online_max_wait": 10_000,
                                                         "noop_rule_min_cases": 6})
    try:
        b = board(0, [(2, y, 4) for y in range(8)] + [(9, 9, 6)])
        moved = b.copy()
        moved[9, 9], moved[9, 10] = 0, 6
        g.new_level(1, b, NO_MASK)
        g.observe(b, click(2, 0), b, changed=False)
        g.observe(b, click(9, 9), moved, changed=True)     # the wait ends at move 2, with 1 case: too few
        assert g.active == {} and g.not_before == {3: 4} and 3 in g.pending
        for y in range(1, 8):
            g.observe(b, click(2, y), b, changed=False)
        assert g.active == {3: {}}, g.events
        kinds = [e["kind"] for e in g.events]
        assert kinds.count("rule not re-bound") >= 1 and kinds[-1] == "rule re-bound"
    finally:
        g.close()


# ---- the map's part ------------------------------------------------------------------------------
class FakeGuide:
    level, train_level, active, rules_version = 2, 1, {1: {}}, 0

    def covered(self, raw, action):
        return ("rule 1",) if action == 2 else ()


def test_gbmap_counts_buttons_a_rule_covers_as_tried():
    m = S.GbMap(FakeGuide())
    b = board(0, [(10, 10, 7)])
    m.observe(b, NO_MASK, False, [1, 2, 3, 4])
    k = m.cur
    assert m._untried_buttons(k) == {0, 1, 3}               # buttons 0-3 available; the rule covers 2
    m.guide.level = 1                                       # on the training level: no map effect
    m.clear()
    m.observe(b, NO_MASK, False, [1, 2, 3, 4])
    assert m._untried_buttons(m.cur) == {0, 1, 2, 3}


def test_gbmap_keeps_the_maps_settings_and_upgrades():
    from upgrades import UpgradedReturnMap
    base = UpgradedReturnMap(stall=123, click_tries=7, gated=True)
    m = S.GbMap.wrap(FakeGuide(), base)
    assert (m.stall, m.click_tries, m.gated, m.bars) == (123, 7, True, None)


# ---- the online bookkeeping ----------------------------------------------------------------------
def test_online_starts_a_level_when_the_score_changes_and_hands_over_the_last_board():
    g = S.Guide("toy")
    o = S.Online(g)
    first = board(0, [(1, 1, 5), (1, 2, 5)])
    o.step(None, None, first, 0, None)
    assert g.level == 1
    nxt = board(0, [(1, 1, 5)])
    o.step(first, 0, nxt, 0, None)
    assert g.moves_on_level == 1
    last = board(0)                                         # colour 5 gone before the winning move
    o.step(nxt, 0, last, 0, None)
    o.step(None, None, board(0, [(4, 4, 5)]), 1, None)      # level 2 starts
    assert g.level == 2 and g.hints == {5: 0}


# ---- the replay ------------------------------------------------------------------------------------
def test_replay_counts_waste_and_masking_from_level_2(tmp_path):
    """Level 1: 25 dead clicks on colour 3. Level 2: 10 more (waste), then one that
    changes something (masking) and 5 after it (no longer covered)."""
    b = board(0, [(5, 5, 3)])
    lit = b.copy()
    lit[20, 20] = 9
    F, N, A, L = [], [], [], []
    for _ in range(25):
        F.append(b), N.append(b), A.append(click(5, 5)), L.append(0)
    F.append(b), N.append(b), A.append(0), L.append(0)      # the level-completing move
    for i in range(16):
        F.append(b), N.append(lit if i == 10 else b), A.append(click(5, 5)), L.append(1)
    d = tmp_path / "transitions"
    d.mkdir()
    np.savez(d / "shard_00000.npz", frames=np.array(F), next_frames=np.array(N), actions=np.array(A),
             levels=np.array(L), action_nums=np.arange(1, len(A) + 1))
    g = S.Guide("toy")
    rows = S.replay(g, str(d), log=lambda m: None)
    assert [r["level"] for r in rows] == [1, 2]
    assert rows[0]["covered"] == 0
    assert (rows[1]["window"], rows[1]["waste"], rows[1]["masking"]) == (16, 10, 1)
