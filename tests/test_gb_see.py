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
    arr, records = S.see_level(ev)
    s = S.summary(ev, arr)
    assert s["changing"] == int(ev.changed.sum()) == len(records)
    assert s["explained"] == 1.0 and s["closure"] == 1.0                 # slides and recolours only
    assert s["kinds"]["moved"] > 0 and s["kinds"]["recoloured"] > 0 and s["kinds"]["reshaped"] == 0
