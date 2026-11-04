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


# ---- v2: colour no-ops judged at the level's end ----------------------------------------------------
V2 = {"colour_judge": "level_end"}


def clicks_that_also_change(cell, n=25, colour=3):
    """`n` clicks on a colour-`colour` cell, each also changing `cell` (a decoration, say)."""
    b = board(0, [(5, 5, colour)])
    after = b.copy()
    after[cell] = 9
    return b, after, [(b, click(5, 5), after)] * n


def test_v1_never_carries_a_colour_whose_clicks_moved_a_decoration_before_the_mask_knew_it():
    g = S.Guide("toy")
    b, after, moves = clicks_that_also_change((0, 10))
    g.new_level(1, b)
    for m in moves:
        g.observe(*m, changed=True)                         # judged under an empty mask at the time
    deco = np.zeros((64, 64), bool)
    deco[0, 10] = True
    g.new_level(2, b, deco, end_mask=deco)
    assert g.covered(b, click(5, 5)) == ()


def test_v2_carries_it_when_every_cell_its_clicks_changed_is_masked_at_the_levels_end():
    g = S.Guide("toy", settings=V2)
    b, after, moves = clicks_that_also_change((0, 10))
    g.new_level(1, b)
    for m in moves:
        g.observe(*m, changed=True)
    deco = np.zeros((64, 64), bool)
    deco[0, 10] = True
    g.new_level(2, b, deco, end_mask=deco)
    assert g.covered(b, click(5, 5)) == ("colour 3",)
    g.observe(b, click(5, 5), b, changed=True)              # dropping is v1's: at once
    assert g.covered(b, click(5, 5)) == ()


def test_v2_does_not_carry_a_colour_whose_clicks_changed_a_live_cell():
    g = S.Guide("toy", settings=V2)
    b, after, moves = clicks_that_also_change((20, 20))
    g.new_level(1, b)
    for m in moves:
        g.observe(*m, changed=False)                        # whatever it was told then
    deco = np.zeros((64, 64), bool)
    deco[0, 10] = True
    g.new_level(2, b, deco, end_mask=deco)
    assert g.covered(b, click(5, 5)) == ()


def test_v2_runs_the_masks_bars_to_the_edges_first():
    """m0r0's case: the mask's bar on row 63 stops at column 57; a click empties (63, 60)."""
    bar = np.zeros((64, 64), bool)
    bar[63, :58] = True
    for settings, expect in ((V2, ("colour 3",)), ({}, ())):
        g = S.Guide("toy", settings=settings)
        b, after, moves = clicks_that_also_change((63, 60))
        g.new_level(1, b, bar)
        for m in moves:
            g.observe(*m)                                   # judged under the level's mask: (63, 60) is live
        g.new_level(2, b, bar)                              # no end mask: the level's own (the replay's case)
        assert g.covered(b, click(5, 5)) == expect


def test_online_hands_the_mask_at_the_level_up_to_v2():
    g = S.Guide("toy", settings=V2)
    o = S.Online(g)
    b, after, _ = clicks_that_also_change((0, 10))
    o.step(None, None, b, 0, None)
    for _ in range(25):
        o.step(b, click(5, 5), after, 0, None)              # no mask yet: each click "changed" something
        o.step(after, 0, b, 0, None)                        # (a button puts it back)
    deco = np.zeros((64, 64), bool)
    deco[0, 10] = True
    o.step(None, None, b, 1, deco)                          # level 2: by now the mask has the decoration
    assert g.colour_noops == {3: 1}


@pytest.mark.parametrize("mask_bars, dropped", [(False, True), (True, False)])
def test_mask_bars_keeps_a_bars_last_cell_from_dropping_a_carried_colour(mask_bars, dropped):
    """Online, the agent's bar on row 63 stops at column 57; a click empties (63, 60)."""
    g = S.Guide("toy", settings=dict(V2, mask_bars=mask_bars))
    o = S.Online(g)
    bar = np.zeros((64, 64), bool)
    bar[63, :58] = True
    b, after, _ = clicks_that_also_change((0, 10))
    o.step(None, None, b, 0, bar)
    for _ in range(25):
        o.step(b, click(5, 5), b, 0, bar)                   # 25 dead clicks on colour 3
    o.step(None, None, b, 1, bar)                           # level 2: colour 3 carried
    assert g.colour_noops == {3: 1}
    emptied = b.copy()
    emptied[63, 60] = 7
    o.step(b, click(5, 5), emptied, 1, bar)
    assert (3 not in g.colour_noops) == dropped


def test_an_unknown_colour_judge_is_refused():
    with pytest.raises(ValueError):
        S.Guide("toy", settings={"colour_judge": "sometimes"})


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
    assert rows[1]["by"] == {"colour": 11} and rows[1]["masking_by"] == {"colour": 1}


def ticking_recording(tmp_path, n_clicks=400):
    """Level 1: `n_clicks` clicks on colour 3 while cell (0, 0) ticks on every move (a
    decoration). Level 2: 30 more clicks. The agent's mask learns (0, 0) only after 200 moves."""
    b0 = board(0, [(5, 5, 3)])
    b1 = b0.copy()
    b1[0, 0] = 1
    F, N, A, L = [], [], [], []
    for i in range(n_clicks + 1 + 30):
        before, after = (b0, b1) if i % 2 == 0 else (b1, b0)
        F.append(before), N.append(after), A.append(click(5, 5) if i != n_clicks else 0)
        L.append(0 if i <= n_clicks else 1)                 # move n_clicks completes level 1
    d = tmp_path / "transitions"
    d.mkdir()
    np.savez(d / "shard_00000.npz", frames=np.array(F), next_frames=np.array(N), actions=np.array(A),
             levels=np.array(L), action_nums=np.arange(1, len(A) + 1))
    return str(d)


@pytest.mark.parametrize("settings, carried", [
    ({}, True),                                              # v1 under S0's mask: the 5a counterfactual
    ({"replay_mask": "agent"}, False),                       # v1 as the agent judged it online: the lag
    ({"replay_mask": "agent", **V2}, True),                  # v2 as the agent would judge it
])
def test_the_replay_with_the_agents_mask_shows_v1s_lag_and_v2_closes_it(tmp_path, settings, carried):
    g = S.Guide("toy", settings=settings)
    rows = S.replay(g, ticking_recording(tmp_path), log=lambda m: None)
    assert [r["level"] for r in rows] == [1, 2]
    assert (3 in g.colour_noops) == carried
    assert rows[1]["masking"] == 0                           # waste and masking: always S0's mask
    assert rows[1]["waste"] == (30 if carried else 0)


def test_the_replay_lists_what_the_online_run_carried(tmp_path):
    corpus = ticking_recording(tmp_path)
    (tmp_path / "glassbox").mkdir()
    with open(tmp_path / "glassbox" / "events.jsonl", "w") as f:
        f.write('{"kind": "start", "level": null, "move": 0}\n')
        f.write('{"kind": "colour no-op carried", "level": 2, "move": 9, "colour": 3, "clicks": 20}\n')
    g = S.Guide("toy")
    S.replay(g, corpus, log=lambda m: None)
    r = {"seed": 0, "events": g.events, "online_carried": S.carried(S.read_events(str(tmp_path / "glassbox" /
                                                                                    "events.jsonl")))}
    assert S.carry_vs_online(r) == "s0: 1 / 1 (0 same level)"   # the same colour, a level later online
    assert S.carry_vs_online({"seed": 0, "events": [], "online_carried": None}) is None
