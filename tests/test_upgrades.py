"""Unit tests for the upgrade screen's module (custom_agents/upgrades.py).

    uv run python -m pytest tests/ -q
"""

import numpy as np
import pytest

from canon import OnlineCanonicalizer
from gridtools import tick_cells, label_components, screen_objects
from upgrades import (Upgrades, UpgradedReturnMap, BarDetector, DEAD_CLICK_TRIES, SCHEDULES, SCHED_STRETCH,
                      OBJCLICK_REDRAW, shape_sizes, ucb_choice)

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


def test_removed_candidates_stop_with_a_pointer_to_the_tag():
    for name in ("graded", "map_objects", "map_diverse"):
        with pytest.raises(SystemExit, match="upgrade-screen-round1"):
            Upgrades([name])


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


def test_bar_mask_reaches_the_map_fingerprint():
    bars = BarDetector()
    bars.sticky[63, :] = True
    m = UpgradedReturnMap(stall=10**9, bars=bars)
    a, b = screen(0), screen(0)
    a[63, 1] = 7
    b[63, 9] = 7
    assert m.key(a, NO_MASK) == m.key(b, NO_MASK)


def gate_sequence(n=120, seed=0):
    m = UpgradedReturnMap(stall=10**9, gated=True)
    rng = np.random.RandomState(seed)
    seq = []
    for _ in range(n):
        m.attempt_steps = int(rng.randint(5, 60))
        m.attempt_new = int(rng.randint(0, 8)) if m.gate_arm == "walk" else int(rng.randint(0, 6))
        m.on_game_over()
        seq.append(m.gate_arm[0])
    return "".join(seq)


def rle(seq):
    """'wwwss' -> '3w 2s': a long run of choices in a form that can be read."""
    out, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        out.append(f"{j - i}{seq[i]}")
        i = j
    return " ".join(out)


def test_the_adopted_gate_chooses_as_it_did_before_map_sched_was_added():
    """map_gated's choices on two made-up reward streams, recorded from the code
    before map_sched was added (16 Nov 2026) and before its UCB formula was moved
    into ucb_choice. map_gated is part of the adopted agent: it must not move."""
    assert rle(gate_sequence()) == "3w 2s 14w 1s 6w 1s 6w 1s 4w 1s 8w 1s 14w 1s 46w 1s 10w"
    assert rle(gate_sequence(seed=3)) == "3w 2s 12w 1s 1w 1s 8w 10s 1w 1s 5w 1s 1w 1s 1w 1s 6w 1s 63w"


# ---- map_sched: a bandit over five map schedules ------------------------------------------------------
def test_the_schedule_table_is_the_one_in_the_plan():
    assert SCHEDULES == {"stay200": (200, 0.0, False), "walk200": (200, 0.5, False),
                         "stay30": (30, 0.0, False), "walk30": (30, 0.5, False), "sweep30": (30, 0.8, True)}


def test_map_sched_takes_map_gateds_place_and_needs_the_map(monkeypatch):
    with pytest.raises(SystemExit, match="one of them"):
        Upgrades(["map_gated", "map_sched"])
    u = Upgrades(["bars", "map_sched"])
    with pytest.raises(SystemExit):
        u.check(map_on=False)
    u.check(map_on=True)
    monkeypatch.setenv("EVAL_GB", "1")              # GbMap would drop the schedule without saying so
    with pytest.raises(SystemExit, match="GbMap"):
        u.check(map_on=True)
    m = u.wrap_map(UpgradedReturnMap(stall=200))
    assert m.sched and not m.gated and m.bars is u.bars
    assert not Upgrades(["bars", "map_gated"]).wrap_map(UpgradedReturnMap(stall=200)).sched


def end_stretch(m, steps, new):
    """A stretch of `steps` decisions that found `new` screens, ended by a game over."""
    m.sched_steps, m.sched_new = steps, new
    m.on_game_over()


def test_sched_starts_on_stay200_and_tries_each_schedule_twice_in_turn():
    m = UpgradedReturnMap(stall=10**9, sched=True)
    assert m.sched_arm == "stay200" and m.stall == 200
    arms = [m.sched_arm]
    for _ in range(9):
        end_stretch(m, 10, 1)
        arms.append(m.sched_arm)
    assert arms == list(SCHEDULES) * 2
    assert all(m.stats["sched_pulls_" + name] == 2 for name in SCHEDULES)


def test_sched_follows_the_schedule_that_finds_more_screens():
    m = UpgradedReturnMap(stall=10**9, sched=True)
    for _ in range(80):
        end_stretch(m, 10, 5 if m.sched_arm == "walk30" else 0)
    pulls = {name: m.stats["sched_pulls_" + name] for name in SCHEDULES}
    assert pulls["walk30"] > 40 and all(pulls[n] >= 2 for n in SCHEDULES)
    assert m.summary()["sched_mean_walk30"] == 0.5


def test_sched_uses_the_gates_formula():
    """After the first tries the choice is ucb_choice's, the formula map_gated uses."""
    m = UpgradedReturnMap(stall=10**9, sched=True)
    for i in range(10):
        end_stretch(m, 10, i % 4)
    assert m.sched_arm == ucb_choice(m.sched_rewards)
    assert all(len(v) == 2 for v in m.sched_rewards.values())


def test_each_schedule_sets_its_stall_its_walk_back_and_its_limit():
    for name, (stall, share, _) in SCHEDULES.items():
        m = UpgradedReturnMap(stall=10**9, sched=True)
        m.attempts.extend([40] * 5)                         # a typical attempt: 40 moves
        m._sched_choice = lambda name=name: name
        end_stretch(m, 10, 1)
        assert m.sched_arm == name and m.stall == stall
        assert m.return_pending == (share > 0)              # "no walk back" clears it, as the gate does
        assert m.return_limit() == int(40 * share)
    plain = UpgradedReturnMap(stall=10**9, gated=True)
    plain.attempts.extend([40] * 5)
    assert plain.return_limit() == 20 and plain.stall == 10**9      # without map_sched: as before


def test_a_stretch_also_ends_after_100_decisions_without_a_game_over():
    m = UpgradedReturnMap(stall=10**9, sched=True)
    has_prev = False
    for i in range(SCHED_STRETCH):
        m.observe(screen(i % 5), NO_MASK, has_prev, [1, 2])     # 5 screens: 5 new ones in 100 decisions
        m.record(0)
        has_prev = True
    assert list(m.sched_rewards["stay200"]) == [5 / SCHED_STRETCH]
    assert m.sched_arm == "walk200" and m.sched_steps == 0
    assert m.stats["sched_decisions_stay200"] == SCHED_STRETCH and m.stats["sched_new_stay200"] == 5
    m.on_game_over()                    # straight after the choice: nothing to credit, the schedule stays
    assert m.sched_arm == "walk200" and not m.sched_rewards["walk200"] and m.return_pending


def test_sweep_presses_each_untried_button_once_and_never_forbids_a_move():
    m = UpgradedReturnMap(stall=10**9, sched=True, rng=np.random.RandomState(0))
    m.sched_arm = "sweep30"
    pressed, has_prev = [], False
    for _ in range(6):
        m.observe(screen(0), NO_MASK, has_prev, [1, 2, 3, 4])   # every button leaves the screen as it is
        a = m.choose()
        if a is None:
            break
        pressed.append(a)
        m.record(a)
        has_prev = True
    assert sorted(pressed) == [0, 1, 2, 3] and m.stats["untried_first"] == 4
    assert m.choose() is None                                   # all pressed: Goose picks, any move allowed
    assert m.override(2, None, None) == (2, None, None)


def test_only_sweep_presses_untried_buttons_first():
    for name in ("stay200", "walk200", "stay30", "walk30"):
        m = UpgradedReturnMap(stall=10**9, sched=True)
        m.sched_arm = name
        m.observe(screen(0), NO_MASK, False, [1, 2, 3, 4])
        assert m.choose() is None and m.stats["untried_first"] == 0
    plain = UpgradedReturnMap(stall=10**9, gated=True)
    plain.observe(screen(0), NO_MASK, False, [1, 2, 3, 4])
    assert plain.choose() is None and "untried_first" not in plain.stats


def test_a_route_step_goes_before_an_untried_button_and_is_still_checked():
    """sweep30 on a known 3-screen line: a stall route walks to the frontier with its
    steps checked as always; a step that lands elsewhere drops the route."""
    m = UpgradedReturnMap(stall=10**9, sched=True, rng=np.random.RandomState(0))
    walk(m, [0, 1, 2], avail=(1,), action=0)                    # 0 -> 1 -> 2 by button 0, the only button
    m.sched_arm = "sweep30"
    m.observe(screen(0), NO_MASK, False, [1])                   # back at the start (a reset)
    m.since_new = m.stall                                       # stalled: route to the frontier (screen 2)
    assert m.choose() == 0 and m.route                          # the route's first step, not an untried press
    m.record(0)
    m.observe(screen(4), NO_MASK, True, [1])                    # the game went somewhere else
    assert not m.route and m.stats["aborted_mismatch"] == 1


# ---- objclick: clicks chosen by shape -----------------------------------------------------------------
def test_shape_sizes_counts_each_shape_and_the_masked_cells_as_one():
    f = np.zeros((64, 64), dtype=np.uint8)
    f[10:12, 10:12] = 3                             # a 2x2 object
    f[30, 30] = 5                                   # a single cell
    f[63, ::2] = 7                                  # a bar's cells, under the mask
    mask = np.zeros((64, 64), dtype=bool)
    mask[63, :] = True
    size, n = shape_sizes(f, mask)
    size = size.reshape(64, 64)
    assert n == 4                                   # background, object, single cell, the masked cells
    assert size[10, 10] == size[11, 11] == 4 and size[30, 30] == 1
    assert size[0, 0] == 4096 - 64 - 4 - 1 and (size[63] == 64).all()
    assert shape_sizes(f, NO_MASK)[1] == 3 + 32                 # unmasked: each of the bar's 32 cells is a shape


def click_probs(value=1.0):
    p = np.zeros(5 + 4096, dtype=np.float32)
    p[5:] = value
    return p


def one_object_screen():
    f = np.zeros((64, 64), dtype=np.uint8)
    f[10, 10] = 9                                   # one small object on a big background
    return f


def test_objclick_draws_shapes_by_mean_score_not_by_size():
    u = Upgrades(["objclick"], seed=3)
    f = one_object_screen()
    u.observe(None, f, NO_MASK, None)
    n, on_object = 2000, 0
    for _ in range(n):
        a, (y, x), c = u.override(5, (40, 40), 64 * 40 + 40, click_probs(), f)
        assert a == 5 and c == 64 * y + x
        on_object += (y, x) == (10, 10)
    # two shapes with the same mean score: half of the redrawn clicks each (Goose alone: 1 in 4,096)
    assert 0.38 * n < on_object < 0.52 * n
    s = u.summary()
    assert s["objclick_redrawn"] + s["objclick_kept"] == n
    assert abs(s["objclick_kept"] / n - (1 - OBJCLICK_REDRAW)) < 0.03
    assert s["objclick_mean_shapes"] == 2.0 and s["objclick_screens"] == 1      # labelled once, then cached


def test_objclick_weights_a_cell_by_its_own_score_inside_a_shape():
    u = Upgrades(["objclick"], seed=0)
    f = np.zeros((64, 64), dtype=np.uint8)          # one shape: the whole screen
    u.observe(None, f, NO_MASK, None)
    p = click_probs(0.0)
    p[5 + 64 * 7 + 9] = 0.75                        # the only cell with any score
    for _ in range(50):
        a, coords, c = u.override(5, (1, 1), 65, p, f)
        assert coords in ((7, 9), (1, 1))           # redrawn there, or Goose's cell kept
    assert u.summary()["objclick_moved"] == u.summary()["objclick_redrawn"] > 30


def test_objclick_leaves_buttons_and_gooses_random_stream_alone():
    u = Upgrades(["objclick"], seed=0)
    f = one_object_screen()
    u.observe(None, f, NO_MASK, None)
    before = u.objclick.rng.bit_generator.state
    assert u.override(2, None, None, click_probs(), f) == (2, None, None)
    assert u.objclick.rng.bit_generator.state == before         # a button draws nothing
    np.random.seed(11)
    for _ in range(100):
        u.override(5, (40, 40), 64 * 40 + 40, click_probs(), f)
    after = np.random.random()
    np.random.seed(11)
    assert after == np.random.random()                          # Goose's stream was not used


def test_objclick_keeps_gooses_cell_when_there_is_nothing_to_draw_from():
    u = Upgrades(["objclick"], seed=0)
    f = one_object_screen()
    u.observe(None, f, NO_MASK, None)
    nan = click_probs()
    nan[5:] = np.nan
    for probs in (click_probs(0.0), nan):
        for _ in range(20):
            assert u.override(5, (40, 40), 64 * 40 + 40, probs, f) == (5, (40, 40), 64 * 40 + 40)
    assert u.summary()["objclick_redrawn"] == 0 and u.summary()["objclick_no_weights"] > 0


def test_objclick_uses_the_mask_of_the_decision_and_the_same_seed_gives_the_same_clicks():
    f = one_object_screen()
    f[63, ::2] = 7
    bar = np.zeros((64, 64), dtype=bool)
    bar[63, :] = True
    picks = []
    for seed in (5, 5, 6):
        u = Upgrades(["objclick"], seed=seed)
        u.observe(None, f, bar, None)
        assert u.objclick.mask is bar
        picks.append([u.override(5, (40, 40), 64 * 40 + 40, click_probs(), f)[2] for _ in range(40)])
    assert picks[0] == picks[1] != picks[2]
    assert u.summary()["objclick_mean_shapes"] == 3.0           # background, object, the bar as one shape


def test_objclick_goes_before_deadclick_and_off_means_untouched():
    u = Upgrades(["objclick", "deadclick"], seed=1, rng=np.random.RandomState(0))
    f = one_object_screen()
    u.observe(None, f, NO_MASK, None)
    for _ in range(DEAD_CLICK_TRIES):
        u.observe(f, f, NO_MASK, 5 + 64 * 10 + 10)              # clicking the object does nothing
    assert (10, 10, 9) in u.blocked
    for _ in range(200):                                        # objclick sends half its clicks there;
        assert u.override(5, (40, 40), 64 * 40 + 40, click_probs(), f)[1] != (10, 10)   # deadclick redraws them
    assert u.stats["dead_click_resamples"] > 50
    off = Upgrades(["bars", "attempt"])
    assert off.objclick is None and off.override(5, (40, 40), 64 * 40 + 40, click_probs(), f) == (5, (40, 40), 64 * 40 + 40)
    assert "objclick_settings" not in off.config() and "map_sched_settings" not in off.config()
    assert Upgrades(["objclick"]).config()["objclick_settings"] == {"redraw": OBJCLICK_REDRAW}
