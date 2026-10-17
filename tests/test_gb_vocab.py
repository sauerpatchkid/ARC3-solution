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
