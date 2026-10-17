"""H1 VOCAB (custom_agents/gb/h1_vocab.py) on hand-made boards: the sandbox finds
the same subject as S1's labelled objects, works out the same facts S1 records,
and is reachable from rule code through gb/core/contract.py."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from gb import h1_vocab as H  # noqa: E402
from gb.core import contract as C  # noqa: E402
from gb.core.evidence import Act, LevelEvidence  # noqa: E402
from test_gb_see import BG, LIVE, board, wall_level  # noqa: E402


def random_board(seed):
    """Blobs of a few colours on black, some touching, one large floor."""
    rng = np.random.default_rng(seed)
    b = board()
    b[30:62, 2:40] = 2                                      # a floor of 1,216 cells: large
    for _ in range(40):
        y, x = rng.integers(0, 60, 2)
        h, w = rng.integers(1, 5, 2)
        b[y:y + h, x:x + w] = rng.choice([3, 4, 9, 12])
    return b


def test_components_are_s1s_objects():
    for seed in range(5):
        b = random_board(seed)
        objs = H.objects(b, LIVE, BG)
        for c in np.unique(b):
            if c == BG:
                continue
            mine = H.components(LIVE & (b == c))
            theirs = [o.cells for o in objs if o.colour == c]     # both in reading order of the first cell
            assert len(mine) == len(theirs) and all(np.array_equal(m, t) for m, t in zip(mine, theirs))


def test_the_subject_of_a_click_is_the_object_under_it():
    b = random_board(0)
    objs = H.objects(b, LIVE, BG)
    for cell in range(0, 64 * 64, 7):
        s = H.subject(b, 5 + cell, LIVE, BG, None)
        under = next((o for o in objs if cell in set(o.cells.tolist())), None)
        assert (s is None) == (under is None)
        if s is not None:
            assert np.array_equal(s.cells, under.cells) and s.colour == under.colour and s.large == under.large


def test_the_subject_of_a_button_is_the_largest_mover_and_a_tie_goes_to_the_first():
    b = board()
    b[10:12, 40:42] = 9                                     # 4 cells, later in reading order
    b[5:7, 50:52] = 9                                       # 4 cells, first in reading order
    b[30, 3] = 9                                            # 1 cell
    s = H.subject(b, 2, LIVE, BG, 9)
    assert s.size == 4 and s.box == (5, 6, 50, 51)
    assert H.subject(b, 2, LIVE, BG, None) is None and H.subject(b, 2, LIVE, BG, 7) is None


def test_a_click_on_a_decoration_cell_has_no_subject():
    b = board()
    b[0, 0:4] = 12
    mask = np.zeros((64, 64), bool)
    mask[0, :] = True
    assert H.subject(b, 5 + 2, ~mask, BG, None) is None


def test_the_sandbox_works_out_every_fact_s1_records():
    ev, arr, records, role, mat, names, feats = wall_level()
    for k in range(len(ev.actions)):
        board_, act, api = ev.inputs(k)
        on = H.features(board_, act, api, role["mover"], with_inside=True)
        assert on == {n for n, v in zip(names, feats[k]) if v}, k


def test_the_sandbox_works_out_clicks_too():
    rng = np.random.default_rng(1)
    b = random_board(1)
    acts = rng.integers(0, 64 * 64, 200) + 5
    f = np.repeat(b[None], len(acts), axis=0)
    ev = LevelEvidence.from_moves("clicks", 0, f, f.copy(), acts.astype(np.int32), mask=np.zeros((64, 64), bool))
    from gb import s1_see
    arr, *_ = s1_see.see_level(ev)
    names = [str(x) for x in arr["feature_names"]]
    feats = np.unpackbits(arr["features"], axis=1)[:, :len(names)].astype(bool)
    for k in range(len(ev.actions)):
        board_, act, api = ev.inputs(k)
        assert H.features(board_, act, api, None, with_inside=True) == {n for n, v in zip(names, feats[k]) if v}


def test_without_with_inside_subject_inside_is_never_set():
    b = board()
    b[10:40, 10:40] = 3
    b[20:22, 20:22] = 9
    api = LevelEvidence.from_moves("x", 0, b[None], b[None], np.array([5 + 20 * 64 + 20], np.int32),
                                   mask=np.zeros((64, 64), bool)).api
    act = Act(5 + 20 * 64 + 20)
    assert "subject_inside" in H.features(b, act, api, None, with_inside=True)
    assert "subject_inside" not in H.features(b, act, api, None)


def test_a_rule_can_call_features():
    src = '''
RULE = "ACTION3 does nothing when the mover touches something on its left"
PARAMS = {"mover": Colour()}
def applies(board, act, api, p):
    return act.action == 3 and "left_touch" in features(board, act, api, p["mover"])
def predict(board, act, api, p):
    return board
'''
    rule = C.Rule(src)
    ev, *_ , role, mat, names, feats = wall_level()
    p = rule.values({"mover": role["mover"]}, ev.api.layout)
    col = names.index("left_touch")
    for k in range(len(ev.actions)):
        board_, act, api = ev.inputs(k)
        assert rule.applies(board_, act, api, p) == bool(int(ev.actions[k]) == 2 and feats[k, col])


def test_a_mover_between_two_walls_sits_on_the_floor():
    # Fixed 16 Oct (S1 VERSION 3): a 2x2 mover between two grey (4) walls on
    # black (5) has as much wall as floor around it. The tie used to go to the
    # lower colour number, so it "sat on" grey and touched nothing. Now a tie goes
    # to the background, whatever the walls' number.
    b = board()
    b[20:22, 10] = b[20:22, 13] = 4
    b[20:22, 11:13] = 9
    s = H.subject(b, 2, LIVE, BG, 9)
    for wall in (4, 8):
        b[20:22, 10] = b[20:22, 13] = wall
        rel, under = H.relations(b, s, LIVE, BG)
        assert under == BG and rel["left"]["touch"] and rel["right"]["touch"]


def test_known_limit_a_mover_in_a_dead_end_sits_on_the_walls():
    # KNOWN LIMIT: what an object sits on is read from the colours around it, so
    # in a dead end (walls on three sides: 6 wall cells, 2 floor) the mover sits
    # on the walls, and they are not obstacles. A floor that is not the
    # background looks the same from one screen, so this is not decidable here.
    b = board()
    b[20:22, 10] = b[20:22, 13] = 4
    b[22, 11:13] = 4
    b[20:22, 11:13] = 9
    rel, under = H.relations(b, H.subject(b, 2, LIVE, BG, 9), LIVE, BG)
    assert under == 4 and not rel["left"]["touch"] and not rel["down"]["touch"]


def test_features_are_fast_on_a_busy_board():
    b = random_board(2)
    api = LevelEvidence.from_moves("x", 0, b[None], b[None], np.array([2], np.int32),
                                   mask=np.zeros((64, 64), bool)).api
    acts = [Act(2)] + [Act(5 + c) for c in range(0, 4096, 97)]
    t0 = time.perf_counter()
    for a in acts:
        H.features(b, a, api, 9)
    ms = (time.perf_counter() - t0) * 1000 / len(acts)
    assert ms < 2.5, f"{ms:.2f} ms per move (a rule may take 5 ms in all)"


# ---- H1's primitives ---------------------------------------------------------------------------------
def api_for(b, mask=None):
    mask = np.zeros((64, 64), bool) if mask is None else mask
    return LevelEvidence.from_moves("x", 0, b[None], b[None], np.array([0], np.int32), mask=mask).api


def test_finding_objects():
    b = board()
    b[10:12, 10:12] = 9
    b[30, 30:40] = 4
    api = api_for(b)
    objs = H.rule_objects(b, api)
    assert [o.colour for o in objs] == [4, 9] and objs[0].size == 10      # largest first
    me = H.find_object(b, api, 9)
    assert me.box == (10, 11, 10, 11) and me.mask.sum() == 4 and not me.mask.flags.writeable
    assert H.find_object(b, api, 7) is None and H.find_object(b, api, BG) is None
    assert H.object_at(b, api, (30, 35)).colour == 4 and H.object_at(b, api, (0, 0)) is None
    assert H.object_at(b, api, None) is None and H.object_at(b, api, (99, 0)) is None
    assert H.rule_objects(board(), api) == []


def test_rays_and_blocking_at_the_border_and_against_a_wall():
    b = board()
    b[10:12, 0:2] = 9                                   # at the left edge
    b[10:12, 5] = 4                                     # a wall 3 cells to its right
    api = api_for(b)
    me = H.find_object(b, api, 9)
    assert H.ray(b, api, (10, 1), 0, 1) == (4, 4) and H.ray(b, api, (10, 0), 0, -1) == (None, None)
    assert H.is_blocked(b, api, me, 0, -1)              # off the screen
    assert not H.is_blocked(b, api, me, 0, 3) and H.is_blocked(b, api, me, 0, 4)
    assert not H.is_blocked(b, api, me, 0, 4, blockers=[8]) and H.is_blocked(b, api, me, 0, 4, blockers=[4])
    assert H.is_blocked(b, api, None, 0, 1)


def test_moving_objects():
    b = board()
    b[10:12, 10:12] = 9
    b[12, 10:12] = 12                                   # a two-colour sprite: 9 over 12
    b[10:13, 20] = 4
    api = api_for(b)
    sprite = H.rule_objects(b, api, 9)[0].mask | H.rule_objects(b, api, 12)[0].mask
    out, claim = H.translate(b, api, sprite, 0, 1)
    assert (out[10:12, 11:13] == 9).all() and (out[12, 11:13] == 12).all() and (out[10:13, 10] == BG).all()
    assert claim.sum() == 9 and (b[10:12, 10:12] == 9).all()                  # old + new cells; b untouched
    out, claim = H.slide(b, api, sprite, 0, 1)
    assert (out[10:12, 18:20] == 9).all() and out[10, 17] == BG             # stops touching the wall
    out, claim = H.slide(b, api, sprite, 0, -1)
    assert (out[10:12, 0:2] == 9).all()                                     # stops at the screen edge
    out, claim = H.translate(b, api, sprite, 0, -11)                         # off the screen: nothing
    assert np.array_equal(out, b) and not claim.any()
    out, claim = H.translate(b, api, None, 0, 1)
    assert np.array_equal(out, b) and not claim.any()


def test_recolour_swap_unchanged():
    b = board()
    b[5, 5] = 3
    b[5, 7] = 12
    api = api_for(b)
    a, c = H.object_at(b, api, (5, 5)), H.object_at(b, api, (5, 7))
    out, claim = H.recolour(b, api, a, 9)
    assert out[5, 5] == 9 and claim.sum() == 1
    out, claim = H.swap(b, api, a, c)
    assert out[5, 5] == 12 and out[5, 7] == 3 and claim.sum() == 2
    big = np.zeros((64, 64), bool)
    big[0:2, 0:2] = True
    assert not H.swap(b, api, a, big)[1].any()                               # different sizes
    out, claim = H.unchanged(b, a)
    assert np.array_equal(out, b) and claim.sum() == 1


def test_a_rule_written_with_primitives_is_exact_on_the_maze():
    from test_gb_guards import maze
    f, n, a, o = maze(0, 9, 600)
    ev = LevelEvidence.from_moves("maze", 0, f, n, a, mask=np.zeros((64, 64), bool))
    src = '''
RULE = "ACTION3 moves the agent one cell left unless it is blocked"
PARAMS = {"agent": Colour()}
def applies(board, act, api, p):
    return act.action == 3 and find_object(board, api, p["agent"]) is not None
def predict(board, act, api, p):
    me = find_object(board, api, p["agent"])
    if is_blocked(board, api, me, 0, -1):
        return unchanged(board, me)
    return translate(board, api, me, 0, -1)
'''
    rule = C.Rule(src)
    p = rule.values({"agent": 9}, ev.api.layout)
    t0, n_applied = time.perf_counter(), 0
    for k in range(len(ev.actions)):
        board_, act, api = ev.inputs(k)
        if rule.applies(board_, act, api, p):
            n_applied += 1
            pred, claim = rule.predict(board_, act, api, p, ev.live)
            assert (pred[claim] == ev.after[k][claim]).all(), k
    ms = (time.perf_counter() - t0) * 1000 / len(ev.actions)
    assert n_applied > 50 and ms < 1.0, ms


def test_the_card_stays_inside_its_budget():
    assert len(H.CARD) // 3 <= 1500                       # the plan's 1,500-token cap, by rules2's high estimate
    for name in H.PRIMITIVES:
        assert f"{name}(" in H.CARD, name
