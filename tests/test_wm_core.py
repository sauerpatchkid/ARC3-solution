"""Unit tests for the Rulebook's core (custom_agents/wm): sandbox, evidence index,
rule checking and trust levels, the rule book with UNKNOWN, coverage, transfer.
Uses a synthetic game whose true rules are known. No LLM needed.

    uv run python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from wm.check import (EXACT, UNKNOWN, WRONG, book_status, check_rule, coverage, grade,  # noqa: E402
                      pick_book, transfer)
from wm.evidence import Act, LevelEvidence, frozen_moves, group_of, ticker_mask  # noqa: E402
from wm.sandbox import compile_functions, validate  # noqa: E402

NO_MASK = np.zeros((64, 64), bool)


# ---- a synthetic game ----------------------------------------------------------------
def step(board, a):
    """ACTION1 slides the 2x2 colour-3 block one column right unless at the wall;
    ACTION2 does nothing; clicking a colour-5 cell turns its whole 2x2 tile to 6."""
    out = board.copy()
    if a == 0:
        ys, xs = np.nonzero(board == 3)
        if xs.max() < 63:
            out[board == 3] = 0
            out[ys, xs + 1] = 3
    elif a >= 5:
        y, x = divmod(a - 5, 64)
        if board[y, x] == 5:
            y0, x0 = y - y % 2, x - x % 2
            out[y0:y0 + 2, x0:x0 + 2] = 6
    return out


def start(shift=0):
    b = np.zeros((64, 64), np.uint8)
    b[10:12, 2 + shift:4 + shift] = 3
    for k in range(6):
        b[30:32, 4 + 4 * k + shift:6 + 4 * k + shift] = 5
    return b


def play(n=600, seed=0, shift=0):
    rng = np.random.default_rng(seed)
    b = start(shift)
    F, N, A = [], [], []
    for t in range(n):
        if t % 150 == 149:
            b = start(shift)                                  # a "reset"
        r = rng.random()
        if r < 0.45:
            a = 0
        elif r < 0.6:
            a = 1
        else:
            y, x = (30, 4 + 4 * int(rng.integers(0, 6)) + shift) if rng.random() < 0.8 else (50, 50)
            a = 5 + 64 * y + x
        nb = step(b, a)
        F.append(b); N.append(nb); A.append(a)
        b = nb
    return np.array(F), np.array(N), np.array(A, np.int32)


GOOD_MOVE = '''
RULE = "ACTION1 slides the colour-3 block one column right unless it touches the right wall"
def applies(board, act, api):
    return act.action == 1 and (board == 3).any() and np.nonzero(board == 3)[1].max() < 63
def predict(board, act, api):
    out = board.copy()
    ys, xs = np.nonzero(board == 3)
    out[board == 3] = api.background
    out[ys, xs + 1] = 3
    return out
'''
OFF_BY_ONE = GOOD_MOVE.replace("xs + 1] = 3", "xs + 2] = 3").replace("max() < 63", "max() < 62")
GOOD_CLICK = '''
RULE = "clicking a colour-5 tile turns the 2x2 tile to 6"
def applies(board, act, api):
    return act.action == 6 and board[act.click] == 5
def predict(board, act, api):
    out = board.copy()
    y, x = act.click
    y0, x0 = y - y % 2, x - x % 2
    out[y0:y0 + 2, x0:x0 + 2] = 6
    return out
'''
NOOP = '''
RULE = "ACTION2 does nothing"
def applies(board, act, api):
    return act.action == 2
def predict(board, act, api):
    return board.copy()
'''
NOOP_TOO_WIDE = NOOP.replace("act.action == 2", "act.action in (1, 2)")


@pytest.fixture(scope="module")
def ev(tmp_path_factory):
    f, n, a = play()
    e = LevelEvidence.from_moves("toy", 0, f, n, a)
    p = str(tmp_path_factory.mktemp("ev") / "toy_l1.npz")
    e.save(p)
    return e, p


# ---- sandbox ------------------------------------------------------------------------
def test_sandbox_rejects_imports_private_attrs_and_files():
    assert validate("import os\ndef applies(b,a,c): return True\ndef predict(b,a,c): return b")[0] is False
    assert validate("def applies(b,a,c): return b.__class__\ndef predict(b,a,c): return b")[0] is False
    assert validate("def applies(b,a,c): return open('x')\ndef predict(b,a,c): return b")[0] is False
    assert validate("def applies(b,a,c): return np.load('x')\ndef predict(b,a,c): return b")[0] is False
    assert validate("def applies(b, a): return True\ndef predict(b,a,c): return b")[0] is False
    assert validate(GOOD_MOVE)[0] is True
    assert "applies" in compile_functions(GOOD_MOVE)


# ---- evidence -----------------------------------------------------------------------
def test_act_and_groups():
    a = Act(5 + 64 * 30 + 4)
    assert a.action == 6 and a.click == (30, 4) and Act(2).action == 3 and Act(2).click is None
    assert group_of(0, start()) == "ACTION1" and group_of(5 + 64 * 30 + 4, start()) == "click on colour 5 (black)"


def test_index_counts_moves_per_key_and_keeps_majority(ev):
    e, _ = ev
    assert int(e.count.sum()) == 600 and len(e.actions) < 600        # repeated keys collapse
    assert e.info["conflict_keys"] == 0 and (e.agree == e.count).all()
    assert set(e.groups) >= {"ACTION1", "ACTION2"}                     # groups with >= 30 moves
    assert "click on colour 5 (black)" in e.group_moves               # rarer groups are still counted
    assert e.api.background == 0 and e.api.layout[0].colour in (3, 5)


def test_conflicting_outcomes_are_flagged_not_dropped():
    b = start()
    f = np.array([b, b, b])
    n = np.array([step(b, 0), step(b, 0), b])                        # same key, two outcomes
    e = LevelEvidence.from_moves("toy", 0, f, n, np.array([0, 0, 0], np.int32), mask=NO_MASK)
    assert len(e.actions) == 1 and e.count[0] == 3 and e.agree[0] == 2 and e.n_variants[0] == 2
    assert e.info["conflict_keys"] == 1 and e.changed[0]              # the majority outcome is kept


def test_ticker_mask_fills_row_gaps_and_frozen_stretches():
    f = np.zeros((400, 64, 64), np.uint8)
    n = f.copy()
    for t in range(400):
        c = [0, 3, 6, 9][t % 4]                                       # a bar ticking along row 63
        n[t, 63, c] = 1
    m, info = ticker_mask(f, n)
    assert m[63, :10].all() and not m[:63].any() and info["filled_cells"] == 6
    ch = np.r_[np.ones(5, bool), np.zeros(600, bool), np.ones(5, bool)]
    fz = frozen_moves(ch)
    assert fz[5:605].all() and not fz[:5].any() and not fz[605:].any()


# ---- checking and trust -------------------------------------------------------------
def test_exact_rule_is_plan_eligible_and_off_by_one_is_not(ev):
    e, p = ev
    good = grade(check_rule(GOOD_MOVE, p), e, "ACTION1")
    assert good["plan_eligible"] and good["accuracy"] == 1.0 and good["gain"] > 0 and good["missed"] == []
    bad = check_rule(OFF_BY_ONE, p)
    g = grade(bad, e)
    assert not g["admitted"] and g["accuracy"] == 0.0 and len(bad["wrong"]) > 0


def test_noop_rules(ev):
    e, p = ev
    assert grade(check_rule(NOOP, p), e)["known_noop"]
    wide = grade(check_rule(NOOP_TOO_WIDE, p), e)                    # wrong wherever ACTION1 moves
    assert not wide["known_noop"] and not wide["admitted"]


def test_static_and_runtime_failures(ev):
    _, p = ev
    assert check_rule("import os", p)["stage"] == "static"
    crash = "def applies(b,a,c): return True\ndef predict(b,a,c): return b[100]"
    assert check_rule(crash, p)["stage"] == "runtime"


def test_a_rule_touching_a_conflict_key_is_not_plan_eligible():
    b = start()
    f = np.array([b] * 30)
    n = np.array([step(b, 0)] * 29 + [b])
    e = LevelEvidence.from_moves("toy", 0, f, n, np.zeros(30, np.int32), mask=NO_MASK)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.npz")
        e.save(path)
        g = grade(check_rule(GOOD_MOVE, path), e)
    assert g["admitted"] and g["accuracy"] == round(29 / 30, 4) and not g["plan_eligible"]


# ---- the book -----------------------------------------------------------------------
def _cand(src, e, p, group, order):
    r = check_rule(src, p)
    return {"group": group, "stage": r["stage"], "order": order, "code": src, "res": r,
            "grade": grade(r, e, group)}


def test_book_unknown_coverage_and_disagreement(ev):
    e, p = ev
    cs = [_cand(GOOD_MOVE, e, p, "ACTION1", 0), _cand(NOOP, e, p, "ACTION2", 1)]
    book, admitted = pick_book(cs)
    assert len(book) == 2 and len(admitted) == 1
    st = book_status([c["res"] for c in book], len(e.actions))
    cov = coverage(st, e)
    clicks = np.array([g.startswith("click") for g in e.group])
    assert (st[clicks] == UNKNOWN).all()                              # uncovered is UNKNOWN, not "nothing"
    assert 0 < cov["coverage"] < 1 and cov["wrong"] == 0
    full, _ = pick_book(cs + [_cand(GOOD_CLICK, e, p, "click on colour 5 (black)", 2)])
    assert coverage(book_status([c["res"] for c in full], len(e.actions)), e)["coverage"] == 1.0
    # two rules that disagree on a key leave it UNKNOWN
    clash = [check_rule(GOOD_MOVE, p), check_rule(NOOP_TOO_WIDE, p)]
    st2 = book_status(clash, len(e.actions))
    moved = np.array([g == "ACTION1" for g in e.group]) & e.changed
    assert (st2[moved] == UNKNOWN).all() and WRONG not in st2[moved] and EXACT in st2


def test_transfer_to_a_shifted_level(tmp_path):
    f, n, a = play(seed=0)
    f2, n2, a2 = play(seed=1, shift=1)                                # "level 2": everything one column over
    tr = LevelEvidence.from_moves("toy", 0, f, n, a)
    te = LevelEvidence.from_moves("toy", 1, f2, n2, a2)
    p2 = str(tmp_path / "l2.npz")
    te.save(p2)
    res = [check_rule(s, p2) for s in (GOOD_MOVE, GOOD_CLICK, NOOP)]
    t = transfer(book_status(res, len(te.actions)), tr, te)
    assert t["book"] == 1.0 and t["test_coverage"] == 1.0 and t["beats_both"]
    assert t["nothing"] < 1.0 and t["memory"] < 1.0
