"""Unit tests for the Coach's screen summary (legacy/coach_track/coach/history.py).

    uv run python -m pytest tests/ -q
"""
import os
import sys

import numpy as np

TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # legacy/coach_track
ROOT = os.path.dirname(os.path.dirname(TRACK))                               # the repo
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, TRACK)

from coach.history import History, Objects, describe  # noqa: E402

NO_MASK = np.zeros((64, 64), dtype=bool)


def screen():
    """Background 0; a 2x2 colour-2 block at rows 10-11, cols 20-21; two 3x3
    colour-5 blocks (same shape) at (40, 40) and (40, 50); a big colour-1 slab."""
    f = np.zeros((64, 64), np.uint8)
    f[10:12, 20:22] = 2
    f[40:43, 40:43] = 5
    f[40:43, 50:53] = 5
    f[0:4, :] = 1
    return f


def click(y, x):
    return 5 + 64 * y + x


# ---- effects ---------------------------------------------------------------------
def test_describe_no_op_move_and_colour_change():
    f = screen()
    assert describe(f, f, NO_MASK)[0] == "no-op"
    g = f.copy()
    g[10:12, 20:22] = 0
    g[10:12, 23:25] = 2                                   # slid 3 columns right
    pat, detail = describe(f, g, f != g)
    assert pat == "colour 2 moved by (+3, +0)" and "4 cells" in detail
    h = f.copy()
    h[10:12, 20:22] = 7                                   # recoloured in place
    assert describe(f, h, f != h)[0] == "colour 2 -> 7"


# ---- objects ---------------------------------------------------------------------
def test_objects_at_and_ranking():
    o = Objects(screen(), NO_MASK)
    it = o.at(10, 21)
    assert it["colour"] == 2 and it["bbox"] == (20, 10, 21, 11) and it["size"] == 4
    assert o.at(30, 30) is None                           # background
    assert o.items[0]["colour"] == 2                      # smallest + rarest first
    fives = [i for i in o.items if i["colour"] == 5]
    assert len(fives) == 2 and all(i["copies"] == 1 for i in fives)
    for i in o.items:                                     # the target cell is inside its object
        y, x = divmod(i["target"], 64)
        assert o.at(y, x)["label"] == i["label"]


def test_masked_cells_are_not_objects():
    mask = NO_MASK.copy()
    mask[10:12, 20:22] = True
    o = Objects(screen(), mask)
    assert o.at(10, 20) is None and all(i["colour"] != 2 for i in o.items)


# ---- history ---------------------------------------------------------------------
def test_click_credit_new_screens_and_background():
    f = screen()
    g = f.copy()
    g[10:12, 20:22] = 7
    h = History()
    h.observe(f, click(10, 20), g, 0)                     # changed + new screen
    h.observe(g, click(30, 30), g, 0)                     # background no-op (seen screen)
    h.observe(f, click(11, 21), f, 0)                     # same object, no-op
    rec = h.level.clicks[(2, (20, 10, 21, 11))]
    assert rec[:3] == [2, 1, 1] and rec[3]["colour 2 -> 7"] == 1 and rec[3]["no-op"] == 1
    assert h.level.last_new == 1                          # only the first move reached a new screen
    assert h.level.background == [1, 0, 0]
    assert h.level.moves == 3 and len(h.level.seen) == 2  # the first screen f counts as seen


def test_level_change_records_the_win():
    f = screen()
    g = f.copy()
    g[40:43, 40:43] = 6
    h = History()
    h.observe(f, 2, f, 0)                                 # ACTION3, nothing
    h.observe(f, click(41, 41), g, 0)                     # the winning click
    nxt = np.zeros((64, 64), np.uint8)
    h.observe(nxt, 0, nxt, 1)                             # first move of level 2
    w = h.wins[0]
    assert w["level"] == 0 and w["moves"] == 2
    assert "colour-5 object (9 cells, box 40,40-42,42)" in w["move"]
    assert h.level.index == 1 and h.level.moves == 1
    assert "Level 1 was won after 2 moves by clicking a colour-5 object" in h.render(nxt)["text"]


def test_render_is_deterministic_and_maps_ids_to_cells():
    def run():
        h = History()
        f = screen()
        for k in range(30):
            h.observe(f, click(10 + k % 2, 20), f, 0)
            h.observe(f, k % 4, f, 0)
        return h.render(f, available=[0, 1, 2, 3, 5])
    a, b = run(), run()
    assert a == b
    for o in a["objects"]:
        y, x = divmod(o["target"], 64)
        bx0, by0, bx1, by1 = o["bbox"]
        assert bx0 <= x <= bx1 and by0 <= y <= by1
    t = a["text"]
    assert "ACTION1 | 8 | 0% | " in t and "no-op (8 of 8)" in t
    assert "0 | 2 | 20,10-21,11 | 4 | 0 | 30 | 0 | 0 | no-op (30 of 30)" in t


def test_render_cap_drops_unclicked_objects_first():
    f = np.zeros((64, 64), np.uint8)
    for k in range(40):                                   # 40 small single-cell objects
        f[2 + (k // 8) * 6, 2 + (k % 8) * 6] = 1 + k % 15
    h = History()
    h.observe(f, click(2, 2), f, 0)                       # one clicked object
    full = h.render(f, available=[5])
    small = h.render(f, available=[5], max_chars=1200)
    assert len(small["text"]) <= 1200 < len(full["text"])
    assert len(small["objects"]) < len(full["objects"])
    assert any(o["bbox"] == (2, 2, 2, 2) for o in small["objects"])   # the clicked one stays


def test_unused_buttons_say_never_pressed_and_clicks_hidden_without_click():
    f = screen()
    h = History()
    h.observe(f, 0, f, 0)
    t = h.render(f, available=[0, 1])["text"]
    assert "ACTION2 | 0 | - | - | never pressed" in t and "OBJECTS ON SCREEN" not in t
