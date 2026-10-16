"""S1 SEE (custom_agents/gb/s1_see.py) on hand-made boards: what each kind of
change is called, what an object leaves behind, and when a change counts as
explained. One test documents a known limit (a two-colour sprite is two objects)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from gb import s1_see as S  # noqa: E402
from gb.core.evidence import LevelEvidence  # noqa: E402
from test_gb_core import play  # noqa: E402

BG = 5                                         # black, the commonest colour on every board here
LIVE = np.ones((64, 64), bool)


def board():
    return np.full((64, 64), BG, np.uint8)


def see(before, after):
    return S.see_case(before, after, LIVE, BG)


def kinds(r):
    return sorted(it["kind"] for it in r["items"])


def test_a_block_that_moves():
    b, a = board(), board()
    b[10:12, 10:12] = 3
    a[10:12, 11:13] = 3
    r = see(b, a)
    assert kinds(r) == ["moved"] and (r["items"][0]["dy"], r["items"][0]["dx"]) == (0, 1)
    assert r["closure"] and r["unit"] == 1.0 and r["explained"]
    assert "moved (+0, +1), leaving colour 5 black" in S.item_text(r["items"][0])


def test_a_blocked_move_changes_nothing():
    b = board()
    b[10:12, 62:64] = 3
    r = see(b, b.copy())
    assert r["changed"] == 0 and r["items"] == [] and r["explained"]


def test_a_recolour():
    b, a = board(), board()
    b[20:23, 20:23] = 12
    a[20:23, 20:23] = 9
    r = see(b, a)
    assert kinds(r) == ["recoloured"] and r["explained"]
    assert "recoloured 12 -> 9" in S.item_text(r["items"][0])


def test_a_push_moves_two_blocks():
    b, a = board(), board()
    b[10:12, 10:12], b[10:12, 12:14] = 3, 4
    a[10:12, 11:13], a[10:12, 13:15] = 3, 4
    r = see(b, a)
    assert kinds(r) == ["moved", "moved"] and all(it["dx"] == 1 for it in r["items"]) and r["explained"]


def test_one_thing_vanishes_and_another_appears():
    b, a = board(), board()
    b[5:7, 5:7] = 6
    a[40:42, 40:43] = 7
    r = see(b, a)
    assert kinds(r) == ["appeared", "vanished"] and r["explained"]


def test_a_two_colour_sprite_is_two_objects():
    # KNOWN LIMIT (plan 5.1): objects are single-colour, so a sprite of two
    # colours is two objects. Both move rigidly, so the move is still explained.
    b, a = board(), board()
    b[10:12, 10:12], b[12, 10:12] = 3, 4
    a[10:12, 13:15], a[12, 13:15] = 3, 4
    r = see(b, a)
    assert kinds(r) == ["moved", "moved"] and r["explained"]


def test_a_bar_that_shrinks_is_one_shrink():
    b, a = board(), board()
    b[63, :] = 12
    a[63, :60] = 12
    r = see(b, a)
    assert kinds(r) == ["shrank"] and r["items"][0]["after"].size == 60 and r["explained"]


def test_a_move_across_a_large_floor_leaves_the_floor_behind():
    b = board()
    b[10:50, 10:50] = 2                        # a floor of 1,600 cells: large
    b[20:22, 20:22] = 9
    a = b.copy()
    a[20:22, 20:22] = 2
    a[20:22, 21:23] = 9
    r = see(b, a)
    assert kinds(r) == ["moved"] and r["items"][0]["under"] == 2 and r["absorbed"] == 1 and r["explained"]


def test_a_move_inside_a_small_box_leaves_the_box_behind():
    b = board()
    b[38:44, 8:14] = 2                         # a 6x6 grey box: small
    b[40:42, 10:12] = 14
    a = b.copy()
    a[40:42, 10:12] = 2
    a[40:42, 8:10] = 14
    r = see(b, a)
    assert kinds(r) == ["moved"] and r["items"][0]["under"] == 2 and r["explained"]


def test_a_trail_on_a_floor_is_on_top_of_the_floor():
    b = board()
    b[10:50, 10:50] = 2                        # floor
    b[30, 10:20] = 8                           # a red trail, 10 cells
    a = b.copy()
    a[30, 20:25] = 8                           # the trail grows over the floor
    r = see(b, a)
    assert kinds(r) == ["grew"] and r["closure"] and r["explained"]


def test_a_rotation_is_not_explained():
    b, a = board(), board()
    b[10:13, 10] = 3
    b[12, 11] = 3                              # an L
    a[10, 10:13] = 3
    a[11, 10] = 3                              # the same L, turned
    r = see(b, a)
    assert kinds(r) == ["reshaped"] and r["closure"] and r["unit"] < 1 and not r["explained"]


def test_moves_pair_by_the_smallest_displacement():
    b, a = board(), board()
    b[10:12, 10:12] = b[10:12, 20:22] = 3
    a[10:12, 12:14] = a[10:12, 18:20] = 3     # both moved two cells, toward each other
    r = see(b, a)
    assert sorted(it["dx"] for it in r["items"]) == [-2, 2]


def test_too_many_items_is_not_explained():
    b, a = board(), board()
    for i in range(10):
        b[2 + 4 * i, 2:4] = 3 + (i % 2)
        a[2 + 4 * i, 3:5] = 3 + (i % 2)
    r = see(b, a)
    assert len(r["items"]) == 10 and r["unit"] == 1.0 and not r["explained"]


def test_a_whole_level_of_the_toy_game():
    f, n, a = play(600, seed=0)
    # No decoration mask: with the estimated one, the block slides through cells
    # marked as decorations and looks clipped (S1 only sees live cells).
    ev = LevelEvidence.from_moves("toy", 0, f, n, a, mask=np.zeros((64, 64), bool))
    arr, records, role, mat = S.see_level(ev)
    s = S.summary(ev, arr)
    assert s["changing"] == int(ev.changed.sum()) == sum(r["changed"] > 0 for r in records)
    assert len(records) == len(ev.actions)                             # every case has a record and a text
    assert s["explained"] == 1.0 and s["closure"] == 1.0                 # slides and recolours only
    assert s["kinds"]["moved"] > 0 and s["kinds"]["recoloured"] > 0 and s["kinds"]["reshaped"] == 0


# ---- steps 4-8: relations, feature bits, roles, matrix, text ----------------------------------------
def wall_game(n=200, seed=0):
    """A 2x2 colour-9 block on black beside a grey wall. ACTION3 slides it left,
    ACTION4 right, unless something is in the way."""
    rng = np.random.default_rng(seed)
    b = board()
    b[20:30, 10] = 4                           # the wall
    b[24:26, 12:14] = 9                        # the block, one cell from the wall
    F, N, A = [], [], []
    for _ in range(n):
        a = int(rng.choice([2, 3]))
        nb = b.copy()
        ys, xs = np.nonzero(b == 9)
        tgt = xs + (-1 if a == 2 else 1)
        if tgt.min() >= 0 and tgt.max() <= 63 and (b[ys, tgt][~np.isin(tgt, xs)] == BG).all():
            nb[b == 9] = BG
            nb[ys, tgt] = 9
        F.append(b)
        N.append(nb)
        A.append(a)
        b = nb
    return np.array(F), np.array(N), np.array(A, np.int32)


def wall_level():
    f, n, a = wall_game()
    ev = LevelEvidence.from_moves("wall", 0, f, n, a, mask=np.zeros((64, 64), bool))
    arr, records, role, mat = S.see_level(ev)
    names = [str(x) for x in arr["feature_names"]]
    feats = np.unpackbits(arr["features"], axis=1)[:, :len(names)].astype(bool)
    return ev, arr, records, role, mat, names, feats


def test_relations_on_a_hand_made_board():
    b = board()
    b[30:32, 30:32] = 9                        # the object
    b[30:32, 33] = 4                           # one empty column to its right, then grey
    o = next(o for o in S.objects(b, LIVE, BG) if o.colour == 9)
    rel, under = S.relations(b, o, LIVE, BG)
    assert under == BG
    assert rel["right"] == {"dist": 2, "colour": 4, "touch": False, "edge": 32}
    assert rel["left"]["dist"] is None and rel["left"]["edge"] == 30
    b[29, 30] = 4
    o = next(o for o in S.objects(b, LIVE, BG) if o.colour == 9)
    assert S.relations(b, o, LIVE, BG)[0]["up"]["touch"]


def test_inside_finds_the_smallest_container():
    b = board()
    b[10:40, 10:40] = 2
    b[20:30, 20:30] = 3
    b[24:26, 24:26] = 9
    objs = S.objects(b, LIVE, BG)
    o = next(o for o in objs if o.colour == 9)
    assert S.inside(o, objs).colour == 3


def test_roles_find_the_mover_and_the_static_colours():
    role = wall_level()[3]
    assert role["mover"] == 9 and role["movers"] == [9] and 4 in role["static"] and role["counters"] == []


def test_a_blocked_move_shows_the_wall_on_its_side():
    ev, arr, records, role, mat, names, feats = wall_level()
    blocked = [r["k"] for r in records if not r["changed"] and ev.actions[r["k"]] == 2]
    moved_left = [r["k"] for r in records if r["changed"] and ev.actions[r["k"]] == 2]
    assert blocked and moved_left
    col = names.index("left_touch")
    assert feats[blocked, col].all() and not feats[moved_left, col].any()     # a one-feature guard separates them
    assert feats[blocked, names.index("left_colour=4")].all()
    assert (arr["dist_left"][blocked] == 1).all()


def test_the_matrix_flags_mixed_outcomes():
    mat = wall_level()[4]
    left = next(r for r in mat if r["family"] == "ACTION3")
    right = next(r for r in mat if r["family"] == "ACTION4")
    assert left["counts"]["nothing"] >= 1 and left["counts"]["moved"] > 0 and left["entropy"] > 0
    assert right["counts"] == {"moved": right["cases"]} and right["entropy"] == 0


def test_the_facts_text_is_pinned():
    # GOLDEN: S2's prompts will quote this text, so a change to it must be on purpose.
    ev, arr, records, *_ = wall_level()
    blocked = next(r for r in records if not r["changed"] and ev.actions[r["k"]] == 2)
    assert blocked["text"] == [
        "ACTION3.",
        "  The mover (colour 9 blue, 2x2, rows 24-25, cols 11-12): up: clear to the screen edge (24 cells); "
        "down: clear to the screen edge (38 cells); left: touching colour 4 darker gray; "
        "right: clear to the screen edge (51 cells).",
        "  Nothing changed."]
    moved = next(r for r in records if r["changed"] and ev.actions[r["k"]] == 3)
    assert moved["text"][-2:] == [
        "  Object (colour 9 blue, 2x2, rows 24-25, cols 12-13) moved (+0, +1), leaving colour 5 black.",
        "  1 other object did not change. The description explains the whole change."]


def test_the_text_keeps_to_its_line_cap():
    b, a = board(), board()
    for i in range(14):
        b[2 + 4 * i, 2:4] = 3 + (i % 2)
        a[2 + 4 * i, 3:5] = 3 + (i % 2)
    r = see(b, a)
    lines = S.facts_text(0, None, {}, None, None, r, [], 0, max_lines=12)
    assert len(lines) == 12 and lines[-2].startswith("  ... and")
