"""custom_agents/carry.py: colour no-ops and goal hints carried from one level to the next,
and the check that it makes the same moves as Glassbox's S5 guide with no rule book."""
import numpy as np
import pytest

import carry as C
from carry import CLICKS, Carry, goal_colours
from gb import artifacts as A
from gb import s0_mask
from gb import s5_act
from gridtools import GRID, complete_bars
from upgrades import Upgrades

NO_MASK = np.zeros((GRID, GRID), bool)


def click(y, x):
    return CLICKS + GRID * y + x


def board(**cells):
    """A blank board with a few cells set: board(c3=(10, 10)) puts colour 3 at row 10, column 10."""
    b = np.zeros((GRID, GRID), np.uint8)
    for name, (y, x) in cells.items():
        b[y, x] = int(name[1:])
    return b


def play_level(c, first, moves, mask=NO_MASK):
    """Start a level on `first`, then feed `moves`: [(action, board after)]."""
    c.on_level()
    c.step(None, None, first, mask)
    cur = first
    for action, after in moves:
        c.step(cur, action, after, mask)
        cur = after
    return cur


# ---- carried colour no-ops ---------------------------------------------------------------------
def test_a_colour_is_carried_after_enough_dead_clicks_and_dropped_at_a_contradiction():
    c, b = Carry(seed=0), board(c5=(10, 10), c3=(20, 20))
    play_level(c, b, [(click(10, 10), b)] * C.CARRY_MIN_CLICKS)          # 20 clicks on colour 5: nothing
    assert c.colour_noops == {}                                          # only carried at the level's end
    play_level(c, b, [])
    assert c.colour_noops == {5: 1} and c.covers(b, click(10, 10)) and not c.covers(b, click(20, 20))
    assert c.weight(b, click(10, 10)) == C.CARRY_ALPHA and c.weight(b, 2) == 1.0
    c.step(b, click(10, 10), board(c5=(10, 10), c3=(20, 21)), NO_MASK)   # now a click on it moves something
    assert c.colour_noops == {} and [e["kind"] for e in c.events][-1] == "colour no-op dropped"


def test_too_few_clicks_or_a_click_that_changed_a_live_cell_is_not_carried():
    c, b = Carry(seed=0), board(c5=(10, 10), c6=(30, 30))
    moved = board(c5=(10, 10), c6=(30, 31))
    play_level(c, b, [(click(10, 10), b)] * (C.CARRY_MIN_CLICKS - 1)                      # one click short
               + [(click(30, 30), moved)] + [(click(30, 31), moved)] * C.CARRY_MIN_CLICKS)   # colour 6 moved once
    play_level(c, b, [])
    assert c.colour_noops == {}


def test_a_click_that_only_moved_a_decoration_does_not_stop_the_carry():
    """Judged at the level's end, under that moment's mask with its bars run to the edges:
    the click changed cell (63, 60), the last cell of a bar the detector masked up to column 40."""
    c, b = Carry(seed=0), board(c5=(10, 10))
    ticked = b.copy()
    ticked[63, 60] = 9
    bar = NO_MASK.copy()
    bar[63, :41] = True
    play_level(c, b, [(click(10, 10), ticked)] + [(click(10, 10), ticked)] * C.CARRY_MIN_CLICKS)
    c.on_level()
    c.step(None, None, b, c.live_mask(bar))
    assert c.colour_noops == {5: 1}


def test_complete_bars_is_the_rule_of_glassbox_s0():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = rng.random((GRID, GRID)) < 0.02
        m[rng.choice([0, 1, 2, 3, 60, 61, 62, 63]), 5:5 + rng.integers(4, 30)] = True
        m[10:10 + rng.integers(4, 30), rng.choice([0, 1, 2, 3, 60, 61, 62, 63])] = True
        assert np.array_equal(complete_bars(m), s0_mask.complete_bars(m))


# ---- goal hints --------------------------------------------------------------------------------
def test_goal_colours_are_the_ones_that_vanished_or_shrank_most_never_the_background():
    first = np.zeros((GRID, GRID), np.uint8)
    first[5:9, 5:9], first[20:22, 20:22], first[40, 40] = 3, 4, 6        # 16 of colour 3, 4 of colour 4, 1 of 6
    last = first.copy()
    last[5:9, 5:9], last[20, 20] = 0, 0                                  # colour 3 gone, colour 4 down by one
    assert goal_colours(first, last, NO_MASK) == [3, 4]                  # vanished first, then shrank; never 0


def test_a_hint_boosts_clicks_on_its_colour_only_where_one_is_possible_and_is_dropped_when_quiet():
    c = Carry(noops=False, seed=0)
    first = board(c3=(10, 10), c4=(20, 20))
    play_level(c, first, [(click(10, 10), board(c4=(20, 20)))])          # colour 3 vanished before the win
    second = board(c3=(12, 12), c4=(22, 22))
    c.on_level()
    c.step(None, None, second, NO_MASK)
    assert c.hints == {3: 0}
    assert c.weight(second, click(12, 12)) == 1.0 + C.CARRY_GAMMA and c.weight(second, click(22, 22)) == 1.0
    probs = np.full(CLICKS + GRID * GRID, 0.5)
    no_three = board(c4=(22, 22))                                        # no hint-coloured cell on this screen:
    for a in range(5):                                                   # nothing is boosted, nothing is rejected
        assert c.override(a, None, None, probs, no_three) == (a, None, None)
    for _ in range(C.CARRY_HINT_DROP):                                   # 2,000 moves that never touch colour 3
        c.step(second, 0, second, NO_MASK)
    assert c.hints == {} and c.events[-1]["kind"] == "goal hint dropped"


# ---- the weights reach Goose by rejection sampling ---------------------------------------------
def test_override_draws_with_probability_p_times_w():
    c, b = Carry(seed=1), board(c5=(10, 10), c3=(20, 20))
    c.colour_noops = {5: 1}
    probs = np.zeros(CLICKS + GRID * GRID)
    probs[click(10, 10)] = probs[click(20, 20)] = 1.0                    # Goose: half and half
    rng, kept = np.random.default_rng(0), 0
    for _ in range(4000):
        first = click(10, 10) if rng.random() < 0.5 else click(20, 20)
        _, _, coord = c.override(CLICKS, divmod(first - CLICKS, GRID), first - CLICKS, probs, b)
        kept += coord == click(10, 10) - CLICKS
    want = 0.5 * C.CARRY_ALPHA / (0.5 * C.CARRY_ALPHA + 0.5)             # p w / sum(p w) = 0.0909
    assert abs(kept / 4000 - want) < 0.02
    assert c.stats["moves"] == 4000 and c.stats["changed_move"] > 1500


# ---- the same moves as Glassbox's guide with no rule book --------------------------------------
def made_up_run(seed, levels=3, moves=700):
    """A game for the differential test, as a generator of frames: (score, board, mask, sampled move).
    Colour 5 never does anything when clicked (until level 3, where one click on it changes a cell);
    a click on colour 3 erases that cell; every move ticks a bar along row 63 that the mask covers only
    up to column 40. The caller sends back the move actually made."""
    rng = np.random.default_rng(seed)
    mask = NO_MASK.copy()
    mask[63, :41] = True
    probs = np.full(CLICKS + GRID * GRID, 0.5)
    for level in range(levels):
        b = np.zeros((GRID, GRID), np.uint8)
        b[8:16, 8:16], b[30:36, 30:36], b[50, 10:14] = 3, 5, 6
        for t in range(moves):
            a = int(rng.integers(0, CLICKS)) if rng.random() < 0.1 else click(*rng.choice([9, 12, 31, 34, 50, 2], 2))
            made = yield level, b, mask, a, probs
            b = b.copy()
            b[63, (t * 3) % GRID] = 1 + t % 2                               # the bar ticks, also past column 40
            if made >= CLICKS:
                y, x = divmod(made - CLICKS, GRID)
                if b[y, x] == 3:
                    b[y, x] = 0
                elif b[y, x] == 5 and level == 2 and t == 300:
                    b[50, 10] = 0                                           # the contradiction


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_carry_makes_the_moves_of_s5_with_no_rule_book(seed):
    settings = s5_act.settings_from(A.load_config())                       # GB-free's settings: glassbox.toml
    assert (settings["colour_judge"], settings["mask_bars"]) == ("level_end", True)
    guide = s5_act.Online(s5_act.Guide("made-up", settings=settings, seed=seed + 7919))
    mine = Carry(seed=seed)
    run = made_up_run(seed)
    frame, prev, prev_action, score, n = next(run), None, None, None, 0
    while True:
        level, b, mask, a, probs = frame
        if level != score:                                                  # as action.py: the score changed
            mine.on_level()
            prev, prev_action, score = None, None, level
        guide.step(prev, prev_action, b, level, mask)
        mine.step(prev, prev_action, b, mine.live_mask(mask))
        sampled = (a, None, None) if a < CLICKS else (CLICKS, divmod(a - CLICKS, GRID), a - CLICKS)
        theirs, ours = guide.override(*sampled, probs, b), mine.override(*sampled, probs, b)
        assert theirs == ours, f"move {n}: S5 chose {theirs}, carry chose {ours}"
        made = ours[0] if ours[0] < CLICKS else CLICKS + ours[2]
        prev, prev_action, n = b, made, n + 1
        try:
            frame = run.send(made)
        except StopIteration:
            break
    g = guide.guide
    assert mine.colour_noops == g.colour_noops and mine.hints == g.hints
    kinds = lambda events: [(e["kind"], e["level"]) for e in events if e["kind"] != "start"]
    assert kinds(mine.events) == kinds(g.events)
    assert {k: mine.stats[k] for k in ("moves", "draws", "changed_move", "covered_first")} == \
        {k: g.stats[k] for k in ("moves", "draws", "changed_move", "covered_first")}
    assert any(k == "colour no-op carried" for k, _ in kinds(mine.events))    # the run did exercise both parts
    assert any(k == "goal hints" for k, _ in kinds(mine.events))
    assert mine.stats["changed_move"] > 50


# ---- wired into EVAL_UPGRADES ------------------------------------------------------------------
def test_upgrades_builds_carry_only_when_asked_and_leaves_moves_alone_otherwise():
    plain = Upgrades(["bars", "attempt"])
    assert plain.carry is None
    probs = np.full(CLICKS + GRID * GRID, 0.5)
    assert plain.override(5, (3, 4), 3 * GRID + 4, probs, board()) == (5, (3, 4), 3 * GRID + 4)
    both = Upgrades(["bars", "carry_noops", "carry_hints"], seed=3)
    assert both.carry.noops and both.carry.use_hints
    assert Upgrades(["carry_hints"]).carry.noops is False
    assert both.config()["carry_settings"]["alpha"] == C.CARRY_ALPHA
    b = board(c5=(10, 10))
    both.on_level()
    both.observe(None, b, NO_MASK, None)
    for _ in range(C.CARRY_MIN_CLICKS):
        both.observe(b, b, NO_MASK, click(10, 10))
    both.on_level()
    both.observe(None, b, NO_MASK, None)
    assert both.carry.colour_noops == {5: 1}
    assert both.summary()["carry"]["colours_carried"] == 1


def test_what_carry_did_reaches_upgrades_stats_json(tmp_path):
    """The counts and the settings are written side by side at exit (they once shared a key)."""
    import json
    u, b = Upgrades(["carry_noops", "carry_hints"], seed=0), board(c5=(10, 10))
    u.on_level()
    u.observe(None, b, NO_MASK, None)
    for _ in range(C.CARRY_MIN_CLICKS):
        u.observe(b, b, NO_MASK, click(10, 10))
    u.on_level()
    u.observe(None, b, NO_MASK, None)
    u.dump(tmp_path / "upgrades_stats.json")
    out = json.loads((tmp_path / "upgrades_stats.json").read_text())
    assert out["carry"]["colours_carried"] == 1 and out["carry"]["events"][0]["colour"] == 5
    assert out["carry_settings"]["min_clicks"] == C.CARRY_MIN_CLICKS
    assert out["upgrades"] == "carry_hints,carry_noops"
