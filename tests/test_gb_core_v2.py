"""Glassbox core, the part inherited from Rulebook v2 CP1 (custom_agents/gb/core):
per-move evidence, metrics and the gate's interval, the rule contract, the
literal lint, re-binding and the composed book. No GPU, no LLM. Copied from
legacy/rulebook/tests/test_wm_cp1.py on 16 Oct 2026 (imports changed only).

    uv run python -m pytest tests/test_gb_core_v2.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_gb_core import GOOD_MOVE, play  # noqa: E402
from gb.core.check import baseline_memory, check_rule  # noqa: E402
from gb.core.evidence import LevelEvidence, read_level, read_moves, screen_hash  # noqa: E402
from gb.core.metrics import (gate_interval, memory_hash, moves_right, nothing_hash, resample_units,  # noqa: E402
                        v1_book_hash)


# ---- the order of the moves (pre-registration 4.4) ---------------------------------------
def record(d, n=400, over_every=40, flagged=False, level_up_at=None, with_nums=True):
    """One recorded run of the toy game with a game over every `over_every`
    moves. Old style (flagged=False): the move that ended the attempt is not in
    the recording and the action number steps by 3. New style: it is recorded
    with game_over=1 and the number steps by 2 (the reset)."""
    F, N, A = play(n, seed=0)
    lv = np.zeros(n, np.int32)
    if level_up_at is not None:
        lv[level_up_at:] = 1
    ends = np.arange(n) % over_every == over_every - 1          # the last recorded move of each attempt
    nums = 1 + np.arange(n) + np.cumsum(np.r_[False, ends[:-1]]) * (1 if flagged else 2)
    os.makedirs(d)
    for s in range(0, n, 150):
        e = slice(s, s + 150)
        arrays = dict(frames=F[e], next_frames=N[e], actions=A[e], levels=lv[e])
        if with_nums:
            arrays["action_nums"] = nums[e].astype(np.int64)
        if flagged:
            arrays["game_over"] = ends[e].astype(np.uint8)
        np.savez(os.path.join(d, f"shard_{s // 150:05d}.npz"), **arrays)
    return F, N, A


@pytest.mark.parametrize("flagged", [False, True])
def test_attempts_and_t_come_from_game_overs(tmp_path, flagged):
    d = str(tmp_path / "run")
    record(d, n=400, over_every=40, flagged=flagged)
    f, n, a, order = read_moves(d, 0)
    assert len(a) == 400 and np.array_equal(order["attempt"], np.arange(400) // 40)
    assert np.array_equal(order["t"], np.arange(400) % 40)
    assert order["num"][0] == 1 and order["num"][40] - order["num"][39] == (2 if flagged else 3)
    f3, n3, a3 = read_level(d, 0)                               # v1's reader: the same moves, no order
    assert np.array_equal(f3, f) and np.array_equal(n3, n) and np.array_equal(a3, a)


def test_the_level_completing_move_is_dropped_and_a_recording_without_numbers_has_no_order(tmp_path):
    d = str(tmp_path / "run")
    record(d, n=400, over_every=40, level_up_at=250)
    _, _, a, order = read_moves(d, 0)
    assert len(a) == 249 and len(order["num"]) == 249 and order["attempt"][-1] == 6
    _, _, a1, order1 = read_moves(d, 1)
    assert len(a1) == 150 and order1["attempt"][0] == 0 and order1["t"][0] == 0   # numbered within the level
    d2 = str(tmp_path / "bare")
    record(d2, with_nums=False)
    assert read_moves(d2, 0)[3] is None
    ev = LevelEvidence.from_corpus("toy", d2, 0)
    assert set(ev.moves) == {"key", "out"}
    with pytest.raises(ValueError, match="no move order"):
        resample_units(ev)


def test_every_move_keeps_its_case_and_its_own_outcome(tmp_path):
    d = str(tmp_path / "run")
    F, N, A = record(d)
    ev = LevelEvidence.from_corpus("toy", d, 0)
    m = ev.moves
    assert len(m["key"]) == 400 and np.array_equal(np.bincount(m["key"], minlength=len(ev.actions)), ev.count)
    for i in (0, 57, 399):                                      # the move's case is its (screen, action)
        assert np.array_equal(ev.before[m["key"][i]], F[i]) and ev.actions[m["key"][i]] == A[i]
        assert m["out"][i] == screen_hash(N[i], ev.live)
    p = str(tmp_path / "ev.npz")
    ev.save(p)
    back = LevelEvidence.load(p)
    assert all(np.array_equal(back.moves[k], m[k]) for k in ("key", "out", "num", "attempt", "t"))
    capped = LevelEvidence.from_corpus("toy", d, 0, max_keys=10)
    kept = capped.moves["key"] >= 0
    assert len(capped.actions) == 10 and 0 < kept.sum() < 400
    assert np.array_equal(np.bincount(capped.moves["key"][kept], minlength=10), capped.count)


def test_evidence_saved_before_cp1_still_loads(tmp_path):
    F, N, A = play(200, seed=0)
    ev = LevelEvidence.from_moves("toy", 0, F, N, A)
    p = str(tmp_path / "old.npz")
    np.savez_compressed(p, game=ev.game, level=ev.level, first=ev.first, mask=ev.mask, info=np.array("{}"),
                        before=ev.before, after=ev.after, actions=ev.actions, count=ev.count, agree=ev.agree,
                        n_variants=ev.n_variants, frozen=ev.frozen)       # exactly what v1 wrote
    old = LevelEvidence.load(p)
    assert old.moves is None and np.array_equal(old.count, ev.count)
    assert check_rule(GOOD_MOVE, p)["stage"] == "checked"


def test_the_split_shares_the_levels_attempts(tmp_path):
    d = str(tmp_path / "run")
    record(d, n=400, over_every=40)
    fit, score = LevelEvidence.split("toy", d, 0, fit_moves=100)
    assert np.array_equal(fit.moves["attempt"], np.arange(100) // 40)
    assert score.moves["attempt"][0] == 2 and score.moves["t"][0] == 20          # a partial attempt, mid-way
    unit, kind, n_att = resample_units(score)
    assert (kind, n_att, unit.max() + 1) == ("blocks of 50 moves", 8, 6)          # 8 attempts < 20
    assert np.array_equal(np.bincount(unit), [50] * 6)
    d2 = str(tmp_path / "many")
    record(d2, n=400, over_every=10)
    unit, kind, n_att = resample_units(LevelEvidence.split("toy", d2, 0, fit_moves=100)[1])
    assert (kind, n_att, unit.max() + 1) == ("attempts", 30, 30) and np.array_equal(np.bincount(unit), [10] * 30)


# ---- per-move rates and the gate's interval -------------------------------------------------
def conflicted(n_major=30, n_minor=10):
    """One screen + ACTION1, seen to end two ways, plus an ordinary case."""
    b = np.zeros((64, 64), np.uint8)
    a1, a2 = b.copy(), b.copy()
    a1[5, 5], a2[6, 6] = 3, 4
    other = b.copy()
    other[0, 0] = 9
    F = np.array([b] * (n_major + n_minor) + [other] * 20)
    N = np.array([a1] * n_major + [a2] * n_minor + [other] * 20)
    A = np.zeros(len(F), np.int32)
    order = {"num": np.arange(len(F), dtype=np.int64), "attempt": (np.arange(len(F)) // 5).astype(np.int32),
             "t": (np.arange(len(F)) % 5).astype(np.int32)}
    return LevelEvidence.from_moves("toy", 0, F, N, A, mask=np.zeros((64, 64), bool), order=order), a1, a2


def test_a_move_is_judged_on_its_own_outcome_not_the_cases_usual_one():
    ev, a1, a2 = conflicted()
    k = int(ev.moves["key"][0])
    assert ev.n_variants[k] == 2 and ev.count[k] == 40 and ev.agree[k] == 30
    usual = nothing_hash(ev)
    usual[k] = screen_hash(a1, ev.live)                          # predicts the usual outcome
    rare = nothing_hash(ev)
    rare[k] = screen_hash(a2, ev.live)                           # predicts the rarer one
    assert moves_right(ev, usual).sum() == 30 + 20 and moves_right(ev, rare).sum() == 10 + 20
    assert moves_right(ev, nothing_hash(ev)).sum() == 20         # "nothing changes": only the still case


def test_the_hashes_agree_with_v1s_per_case_scoring(tmp_path):
    d = str(tmp_path / "run")
    record(d, n=400, over_every=40, level_up_at=250)
    train = LevelEvidence.from_corpus("toy", d, 0)
    test = LevelEvidence.from_corpus("toy", d, 1)
    usual = np.array([screen_hash(a, test.live) for a in test.after], np.uint64)
    assert np.array_equal(memory_hash(train, test) == usual, baseline_memory(train, test))
    assert np.array_equal(nothing_hash(test) == usual, ~test.changed)
    p = str(tmp_path / "test.npz")
    test.save(p)
    r = check_rule(GOOD_MOVE, p)
    book = v1_book_hash([r], test)
    assert np.array_equal((book == usual)[r["applies"]], r["correct"][r["applies"]])
    assert np.array_equal(book[~r["applies"]], nothing_hash(test)[~r["applies"]])   # UNKNOWN reads as unchanged


def _moves(n_units, per_unit):
    class Ev:
        moves = {"key": np.zeros(n_units * per_unit, np.int32),
                 "attempt": np.repeat(np.arange(n_units), per_unit).astype(np.int32)}
    return Ev


def test_gate_interval_follows_the_registered_recipe():
    ev = _moves(40, 10)
    rng = np.random.default_rng(1)
    nothing = rng.random(400) < 0.3
    memory = nothing | (rng.random(400) < 0.2)
    book = memory | (rng.random(400) < 0.3)                      # at least as good as both on every move
    g = gate_interval(ev, {"book": book, "nothing": nothing, "memory": memory})
    assert (g["unit"], g["units"], g["moves"]) == ("attempts", 40, 400)
    assert g["rate"]["book"] == round(float(book.mean()), 4)
    for b, arm in (("nothing", nothing), ("memory", memory)):
        d = g["diff"][b]
        assert d["point"] == round(float(book.mean() - arm.mean()), 4) and 0 < d["lo"] < d["point"] < d["hi"]
        # the recipe, written out: 2,000 resamples of 40 units, seed 0, pooled moves, 2.5-97.5 percentiles
        draw = np.random.default_rng(0).integers(0, 40, size=(2000, 40))
        per = lambda v: v.reshape(40, 10).sum(axis=1)
        diffs = (per(book)[draw].sum(axis=1) - per(arm)[draw].sum(axis=1)) / 400
        assert (d["lo"], d["hi"]) == tuple(round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5]))
    assert g["beats_both"]
    assert gate_interval(ev, {"book": book, "nothing": nothing, "memory": memory}) == g      # seed 0: repeatable
    tie = gate_interval(ev, {"book": memory, "nothing": nothing, "memory": memory})
    assert tie["diff"]["memory"] == {"point": 0.0, "lo": 0.0, "hi": 0.0, "above_zero": False}
    assert tie["diff"]["nothing"]["above_zero"] and not tie["beats_both"]


def test_units_of_unequal_size_are_pooled_by_moves_not_averaged():
    class Ev:
        moves = {"key": np.zeros(30 * 2 + 300, np.int32),
                 "attempt": np.r_[np.repeat(np.arange(30), 2), np.full(300, 30)].astype(np.int32)}
    book = np.r_[np.ones(60, bool), np.zeros(300, bool)]         # right on the 30 short attempts only
    g = gate_interval(Ev, {"book": book, "nothing": np.zeros(360, bool), "memory": np.zeros(360, bool)})
    assert g["rate"]["book"] == round(60 / 360, 4) and g["units"] == 31
    assert g["diff"]["nothing"]["hi"] == 1.0                     # a resample without the long attempt
    assert g["diff"]["nothing"]["lo"] < 0.1                      # one with it several times


def test_too_few_units_gives_no_interval_and_does_not_count():
    ev = _moves(4, 50)                                           # 4 attempts -> blocks of 50 -> 4 units
    book, none = np.ones(200, bool), np.zeros(200, bool)
    g = gate_interval(ev, {"book": book, "nothing": none, "memory": none})
    assert (g["unit"], g["units"]) == ("blocks of 50 moves", 4)
    assert g["diff"]["nothing"] == {"point": 1.0, "lo": None, "hi": None, "above_zero": False}
    assert not g["beats_both"]
    book, none = np.ones(250, bool), np.zeros(250, bool)        # one more block: 5 units is enough
    assert gate_interval(_moves(5, 50), {"book": book, "nothing": none, "memory": none})["beats_both"]


def test_moves_whose_case_was_dropped_are_left_out_of_the_rates():
    ev = _moves(40, 10)
    ev.moves["key"][:100] = -1
    book, none = np.ones(400, bool), np.zeros(400, bool)
    g = gate_interval(ev, {"book": book, "nothing": none, "memory": none})
    assert (g["moves"], g["moves_without_a_case"], g["rate"]["book"]) == (300, 100, 1.0)


# ---- the rule contract, v2 (plan section 4.1) -----------------------------------------------
from gb.core import contract as C  # noqa: E402
from gb.core.lint import feedback, lint  # noqa: E402

# The toy game's slide as a v2 rule: the block's colour and its step are level facts.
SLIDE = '''
RULE = "ACTION1 slides the agent block right by its step unless it would leave the board"
PARAMS = {"agent": Colour(), "step": Int(1, 3)}
def bind(board0, api):
    return {"agent": min(c for c in api.colours if c != api.background), "step": 1}
def applies(board, act, api, p):
    xs = np.nonzero(board == p["agent"])[1]
    return act.action == 1 and len(xs) > 0 and xs.max() + p["step"] < 64
def predict(board, act, api, p):
    out = board.copy()
    ys, xs = np.nonzero(board == p["agent"])
    out[ys, xs] = api.background
    out[ys, xs + p["step"]] = p["agent"]
    return out, (board == p["agent"]) | (out == p["agent"])
'''


def toy_ev(n=300, **kw):
    F, N, A = play(n, seed=0)
    kw.setdefault("mask", np.zeros((64, 64), bool))             # the toy has no ticker
    return LevelEvidence.from_moves("toy", 0, F, N, A, **kw)


def test_params_are_read_without_running_the_rule():
    assert C.read_params(GOOD_MOVE) == (1, {})
    version, params = C.read_params(SLIDE)
    assert version == 2 and list(params) == ["agent", "step"]
    assert isinstance(params["agent"], C.Colour) and (params["step"].lo, params["step"].hi) == (1, 3)
    assert C.check_form(SLIDE)[0] and C.check_form(GOOD_MOVE)[:3] == (True, "ok", 1)
    v2 = "def applies(board, act, api, p): return True\ndef predict(board, act, api, p): return board\n"
    assert C.read_params(v2) == (2, {})                        # v2 form with nothing to bind
    bad = {
        "fourth argument": 'PARAMS = {"a": Colour()}\n' + v2.replace(", p)", ")"),
        "literal dict": "PARAMS = dict(a=Colour())\n" + v2,
        "must be one of": 'PARAMS = {"a": Shade()}\n' + v2,
        "plain literal arguments": 'PARAMS = {"a": Int(1, len("abc"))}\n' + v2,
        "lo <= hi": 'PARAMS = {"a": Int(5, 1)}\n' + v2,
        "from 1 to 8": 'PARAMS = {"a": Offset(40)}\n' + v2,
        "at most 6": "PARAMS = {" + ", ".join(f'"s{i}": Colour()' for i in range(7)) + "}\n" + v2,
        "bind must take": v2 + "def bind(board0): return {}\n",
        "both take": v2.replace("def predict(board, act, api, p)", "def predict(board, act, api)"),
        "no function named predict": "def applies(board, act, api, p): return True\n",
    }
    for why, src in bad.items():
        with pytest.raises(C.RuleError, match=why):
            C.read_params(src)
    ok, why, _, _ = C.check_form("import os\n" + SLIDE)        # the sandbox's own rules still hold
    assert not ok and "Import" in why


def test_a_levels_bindings_are_plain_data():
    ev = toy_ev()
    _, params = C.read_params('PARAMS = {"c": Colour(), "d": Offset(1), "r": Region(), "k": Choice(["a", (1, 2)])}\n'
                              "def applies(b, a, api, p): return True\ndef predict(b, a, api, p): return b\n")
    dom = C.domains(params, ev)
    assert dom["c"] == [0, 3, 5, 6]                            # first board: 0, 3, 5; colour 6 appears in changes
    assert len(dom["d"]) == 9 and (0, 1) in dom["d"]
    assert dom["r"] == list(range(len(ev.api.layout))) and dom["k"] == ["a", (1, 2)]
    assert C.n_bindings(dom) == 4 * 9 * len(ev.api.layout) * 2
    some = list(C.all_bindings(dom, fixed={"c": 3, "r": 0, "k": "a"}))
    assert len(some) == 9 and some[0] == {"c": 3, "d": (-1, -1), "r": 0, "k": "a"}
    import json
    back = json.loads(json.dumps(some[0]))                     # JSON turns the pair into a list
    assert C.freeze(back) == C.freeze(some[0])
    assert "region 0 (colour" in C.describe(some[0], params, ev.api.layout)


def test_a_rule_is_called_one_way_whatever_its_form():
    ev = toy_ev()
    k = int(np.flatnonzero((ev.actions == 0) & ev.changed)[0])
    board, act, api = ev.inputs(k)
    assert api.level == 1 and api.colours == (0, 3, 5) and api.t is None
    old, new = C.Rule(GOOD_MOVE), C.Rule(SLIDE)
    assert (old.version, old.params, new.version) == (1, {}, 2)
    p = new.values({"agent": 3, "step": 1}, api.layout)
    with pytest.raises(TypeError):
        p["agent"] = 5                                         # read-only
    assert old.applies(board, act, api, None) and new.applies(board, act, api, p)
    whole, claim_all = old.predict(board, act, api, None, ev.live)
    part, claim = new.predict(board, act, api, p, ev.live)
    assert np.array_equal(whole, ev.after[k]) and np.array_equal(part, ev.after[k])
    assert claim_all is ev.live and claim.sum() == 6           # the block's 4 old and 4 new cells, 2 shared
    wrong = new.values({"agent": 5, "step": 1}, api.layout)
    assert not np.array_equal(new.predict(board, act, api, wrong, ev.live)[0], ev.after[k])
    assert new.guess(api) == ({"agent": 3, "step": 1}, None)


def test_bind_answers_are_cleaned_and_a_region_is_handed_over_as_the_region():
    ev = toy_ev()
    api = ev.api
    src = ('PARAMS = {"r": Region(), "c": Colour(), "d": Offset(2), "n": Int(1, 4)}\n'
           "def bind(board0, api):\n    return {\"r\": api.layout[1], \"c\": 99, \"d\": [0, 2], \"n\": 2.5, \"zz\": 1}\n"
           "def applies(board, act, api, p): return p[\"r\"].colour == p[\"c\"]\n"
           "def predict(board, act, api, p): return board\n")
    rule = C.Rule(src)
    assert rule.guess(api) == ({"r": 1, "d": (0, 2)}, None)     # colour 99 and step 2.5 are dropped
    p = rule.values({"r": 1, "c": api.layout[1].colour, "d": [0, 2], "n": 1}, api.layout)
    assert p["r"] is api.layout[1] and p["d"] == (0, 2) and rule.applies(None, None, api, p)
    broken = C.Rule(src.replace('return {"r": api.layout[1]', 'return {"r": api.layout[999]'))
    assert broken.guess(api)[0] == {} and "bind() failed" in broken.guess(api)[1]
    for bad, why in ((("board", "board"), "64x64 array"), ("board[:5]", "64x64 board"), ("(board, board, 1)", "board or")):
        out = bad if isinstance(bad, str) else f"({bad[0]}, {bad[1]}[:3])"
        r = C.Rule(f"def applies(board, act, api, p): return True\ndef predict(board, act, api, p): return {out}\n")
        with pytest.raises(ValueError, match=why):
            r.predict(ev.before[0], None, api, None, ev.live)


# ---- the literal lint (plan section 4.2) -------------------------------------------------------
def kinds(src):
    return [(w["kind"], w["code"]) for w in lint(src)]


def body(lines, name="predict", sig="board, act, api, p"):
    return f"def {name}({sig}):\n" + "".join(f"    {ln}\n" for ln in lines)


def test_lint_flags_level_facts_and_leaves_mechanisms_alone():
    assert lint(SLIDE) == [] and feedback([]) == ""
    v1 = kinds(GOOD_MOVE)                                       # v1's style: the colour is in the code
    assert ("colour literal", "board == 3") in v1 and all(k == "colour literal" for k, _ in v1)
    assert not any("63" in code for _, code in v1)              # the board's edge is not a level fact
    assert kinds(body(["sub = board[1:3]", "return (sub == 7).any()"])) == [("colour literal", "sub == 7")]
    assert kinds(body(["n = (board == p['c']).sum()", "return n == 4"])) == []      # a count, not a colour
    assert kinds(body(["return board[act.click] in (8, 9)"])) == [("colour literal", "board[act.click] in (8, 9)")]
    assert kinds(body(["return any(r.colour == 9 for r in api.layout)"])) == [("colour literal", "r.colour == 9")]
    assert kinds(body(["out = board.copy()", "out[board == p['c']] = 14", "return out"])) == \
        [("colour literal", "out[board == p['c']] = 14")]
    assert kinds(body(["out = board.copy()", "out[board == p['c']] = api.background", "return out"])) == []
    assert kinds(body(["return np.where(board == p['c'], 9, board)"])) == \
        [("colour literal", "np.where(board == p['c'], 9, board)")]
    assert kinds(body(["return act.action == 4 or act.action in (4, 5)"], "applies")) == []
    assert kinds(body(["return board[10:12, 2].any()"])) == [("coordinate literal", "10"), ("coordinate literal", "12")]
    assert kinds(body(["ys, xs = np.nonzero(board)", "return board[ys, xs + 1], board[-1], board[:, 63], board[::2]"])) == []
    assert kinds(body(["return act.click == (30, 4)"], "applies")) == [("coordinate pair", "(30, 4)")]
    assert kinds(body(["return board[30, 4]"])) == [("coordinate pair", "30, 4")]             # once, not three times
    assert kinds(body(["m = np.zeros((64, 64), bool)", "d = (0, 1)", "return m"])) == []
    assert kinds(body(["return {'agent': 3, 'at': (30, 4)}"], "bind", "board0, api")) == []    # bind may guess
    assert kinds("def applies(:\n") == []                       # not this pass's job


def test_lint_feedback_says_what_to_move():
    text = feedback(lint(GOOD_MOVE))
    assert "Move each of these into PARAMS" in text and "`board == 3`" in text and "Colour() slot" in text
    assert all(w["line"] > 1 for w in lint(GOOD_MOVE))


# ---- re-binding by search (plan section 4.3) ----------------------------------------------------
from gb.core import bind as B  # noqa: E402
from gb.core.metrics import claimed_grade  # noqa: E402

NO_MASK = np.zeros((64, 64), bool)
LEVEL1 = dict(agent=3, tile=5, lit=6, step=1)                   # the toy game's level facts
LEVEL2 = dict(agent=9, tile=11, lit=12, step=2)

# Clicking a tile lights it: both colours are level facts. Claims the tile only.
LIGHT = '''
RULE = "clicking a tile turns its 2x2 square to the lit colour"
PARAMS = {"tile": Colour(), "lit": Colour()}
def applies(board, act, api, p):
    return act.action == 6 and board[act.click] == p["tile"] and p["tile"] != p["lit"]
def predict(board, act, api, p):
    out = board.copy()
    y, x = act.click
    y0, x0 = y - y % 2, x - x % 2
    out[y0:y0 + 2, x0:x0 + 2] = p["lit"]
    claimed = np.zeros((64, 64), bool)
    claimed[y0:y0 + 2, x0:x0 + 2] = True
    return out, claimed
'''
NO_BIND = "\n".join(ln for ln in SLIDE.split("\n") if "bind" not in ln and "api.colours" not in ln)
WRONG_WAY = SLIDE.replace('xs + p["step"]] = p["agent"]', 'xs - p["step"]] = p["agent"]')


def play2(n, seed, agent, tile, lit, step, only=None):
    """The toy game with its level facts as arguments: ACTION1 slides the 2x2
    agent block `step` columns right, ACTION2 does nothing, clicking a tile
    lights it. `only`: keep just these kinds of action ('slide', 'wait', 'click')."""
    rng = np.random.default_rng(seed)

    def start():
        b = np.zeros((64, 64), np.uint8)
        b[10:12, 2:4] = agent
        for k in range(6):
            b[30:32, 4 + 4 * k:6 + 4 * k] = tile
        return b
    b = start()
    F, N, A = [], [], []
    for t in range(n):
        if t % 40 == 39:
            b = start()
        r = rng.random()
        kind = "slide" if r < 0.45 else "wait" if r < 0.6 else "click"
        if only and kind not in only:
            continue
        nb = b.copy()
        if kind == "slide":
            a = 0
            ys, xs = np.nonzero(b == agent)
            if xs.max() + step < 64:
                nb[ys, xs] = 0
                nb[ys, xs + step] = agent
        elif kind == "wait":
            a = 1
        else:
            y, x = (30, 4 + 4 * int(rng.integers(0, 6))) if rng.random() < 0.8 else (50, 50)
            a = 5 + 64 * y + x
            if b[y, x] == tile:
                nb[30:32, x:x + 2] = lit
        F.append(b); N.append(nb); A.append(a)
        b = nb
    return np.array(F), np.array(N), np.array(A, np.int32)


def level(tmp_path, name, facts, n=400, seed=0, only=None, mask=NO_MASK, level=0):
    ev = LevelEvidence.from_moves("toy", level, *play2(n, seed, only=only, **facts), mask=mask)
    path = str(tmp_path / f"{name}.npz")
    ev.save(path)
    return ev, path


def test_search_finds_the_levels_binding_and_rebinds_on_the_next(tmp_path):
    ev1, p1 = level(tmp_path, "l1", LEVEL1)
    found = B.search(SLIDE, p1, family="ACTION1")
    assert found["stage"] == "checked" and found["bindings"] == 4 * 3 and found["untried"] == 0
    assert [h["cases"] for h in found["halving"]] == [10, 50]
    assert found["halving"][0]["tried"] == 12 and found["halving"][-1]["exact"] < 12       # the halving cuts
    best, flag = B.choose(found)
    assert best["binding"] == {"agent": 3, "step": 1} and flag is None and best["guess"]
    assert found["exact_applying"] == 1 and found["exact"] == 1 + 3      # colour 0 never applies: exact, and last
    assert found["exact_list"][0]["applies"] == int(best["res"]["applies"].sum()) > 0
    g = claimed_grade(best["res"], ev1)
    assert g["trusted_v2"] and g["claimed_exactness"] == 1.0 and g["cells_right"] >= 20

    ev2, p2 = level(tmp_path, "l2", LEVEL2, seed=1, level=1)
    untouched = B.run_rule(SLIDE, p2, best["binding"])           # T0: level 1's binding on level 2
    assert untouched["stage"] == "checked" and not untouched["applies"].any()
    found2 = B.search(SLIDE, p2, family="ACTION1", prefer=best["binding"])
    best2, flag2 = B.choose(found2)
    assert best2["binding"] == {"agent": 9, "step": 2} and flag2 is None and not best2["guess"]   # bind() said step 1
    assert found2["guess"] == {"agent": 9, "step": 1}
    assert claimed_grade(best2["res"], ev2)["trusted_v2"]


def test_a_two_colour_rule_rebinds_and_a_v1_rule_is_its_own_single_binding(tmp_path):
    ev1, p1 = level(tmp_path, "l1", LEVEL1)
    ev2, p2 = level(tmp_path, "l2", LEVEL2, seed=1, level=1)
    one, _ = B.choose(B.search(LIGHT, p1, family="click"))
    two, _ = B.choose(B.search(LIGHT, p2, family="click", prefer=one["binding"]))
    assert one["binding"] == {"tile": 5, "lit": 6} and two["binding"] == {"tile": 11, "lit": 12}
    assert two["res"]["claimed_n"][two["res"]["applies"]].max() == 4          # it vouches for the tile only

    old = B.search(GOOD_MOVE, p1, family="ACTION1")              # v1 form: nothing to bind
    assert (old["version"], old["bindings"], old["exact"]) == (1, 1, 1)
    best, flag = B.choose(old)
    assert best["binding"] == {} and flag is None
    ref = check_rule(GOOD_MOVE, p1)                               # and it is scored exactly as v1 scored it
    for k in ("applies", "correct", "changes", "phash", "claimed_ok", "cells_right", "cells_false", "claimed_n"):
        assert np.array_equal(best["res"][k], ref[k]), k
    gone, flag = B.choose(B.search(GOOD_MOVE, p2, family="ACTION1"))          # colour 3 is not on level 2
    assert flag == "binding untested" and not gone["res"]["applies"].any()


def test_no_exact_binding_is_a_failed_binding_with_its_mistakes(tmp_path):
    ev1, p1 = level(tmp_path, "l1", LEVEL1)
    found = B.search(WRONG_WAY, p1, family="ACTION1")
    assert found["stage"] == "checked" and found["exact_applying"] == 0
    assert B.choose(found) == (None, "binding failed")           # the bindings that never apply do not count
    tried = [f for f in found["finals"] if f["res"]["applies"].any()]
    assert not any(f["exact"] for f in tried) or not tried
    static = B.search("def applies(board, act, api, p): return True\n", p1)
    assert (static["stage"], static["reason"]) == ("static", "no function named predict")
    crash = B.search(SLIDE.replace('xs.max() + p["step"] < 64', "undefined_name"), p1)
    assert crash["stage"] == "runtime" and "undefined_name" in crash["reason"]


def test_a_rule_that_never_applies_on_the_new_level_keeps_only_a_bind_guess(tmp_path):
    ev, path = level(tmp_path, "quiet", LEVEL2, only=("wait", "click"), level=1)       # no ACTION1 was recorded
    found = B.search(SLIDE, path, family="ACTION1")
    best, flag = B.choose(found)
    assert flag == "binding untested" and best["binding"] == {"agent": 9, "step": 1} and best["guess"]
    assert found["exact_applying"] == 0 and found["exact"] == found["bindings"]
    assert B.choose(B.search(NO_BIND, path, family="ACTION1")) == (None, "binding failed")


def test_too_many_bindings_need_bind_and_unused_slots_are_not_searched(tmp_path):
    ev, path = level(tmp_path, "l1", LEVEL1)
    big = SLIDE.replace('PARAMS = {"agent": Colour(), "step": Int(1, 3)}',
                        'PARAMS = {"agent": Colour(), "step": Int(1, 3), "a": Offset(8), "b": Offset(8)}')
    used = big.replace("return act.action == 1 and", 'return p["a"] != p["b"] and act.action == 1 and')
    spare = B.search(big, path, family="ACTION1")                # a and b are never read
    assert spare["unused"] == ["a", "b"] and spare["bindings"] == 12 * 289 * 289
    assert spare["halving"][0]["tried"] == 12 and B.choose(spare)[0]["binding"]["agent"] == 3
    much = B.search(used.replace("def bind", "def no_bind"), path, family="ACTION1")
    assert much["finals"] == [] and "more than 4,096" in much["reason"] and B.choose(much) == (None, "binding failed")
    fixed = B.search(used.replace('"step": 1}', '"step": 1, "a": (0, 1)}'), path, family="ACTION1")
    assert fixed["halving"][0]["tried"] == 12 * 289              # bind() pinned a; b is searched
    assert B.choose(fixed)[0]["binding"]["a"] == (0, 1)


def test_a_search_out_of_time_never_calls_an_untried_binding_exact(tmp_path):
    ev, path = level(tmp_path, "l1", LEVEL1)
    found = B.search(SLIDE, path, family="ACTION1", budget_s=0.0)
    assert found["stage"] == "checked" and found["untried"] == 12 and found["exact"] == 0
    assert len(found["finals"]) == 1 and not found["finals"][0]["exact"]
    assert B.choose(found) == (None, "binding failed")


def test_region_numbers_belong_to_the_evidence_they_were_bound_on(tmp_path):
    pad = '''
RULE = "clicking inside the pad region lights it"
PARAMS = {"pad": Region(), "lit": Colour()}
def applies(board, act, api, p):
    return act.action == 6 and bool(p["pad"].mask[act.click]) and board[act.click] == p["pad"].colour
def predict(board, act, api, p):
    out = board.copy()
    out[p["pad"].mask] = p["lit"]
    return out, p["pad"].mask
'''
    fit, p_fit = level(tmp_path, "fit", LEVEL1, n=300)
    hide = NO_MASK.copy()
    hide[10:12, 2:4] = True                                       # the other split masks the block's start...
    score, p_score = level(tmp_path, "score", LEVEL1, n=300, seed=3, mask=hide)
    assert len(score.api.layout) == len(fit.api.layout) - 1       # ...so its layout numbers regions differently
    best, _ = B.choose(B.search(pad, p_fit, family="click"))
    i = best["binding"]["pad"]
    assert best["binding"]["lit"] == 6 and fit.api.layout[i].colour == 5
    right = B.run_rule(pad, p_score, best["binding"], layout_path=p_fit)
    naive = B.run_rule(pad, p_score, best["binding"])             # the same number in the other layout
    on = right["applies"]
    assert on.any() and right["claimed_ok"][on].all()
    assert not np.array_equal(naive["applies"], right["applies"])


def test_the_informative_cases_come_first():
    ev = toy_ev()
    order = B.case_order(ev, "ACTION1")
    n_first = int(((ev.actions == 0) & ev.changed).sum())
    assert sorted(order.tolist()) == list(range(len(ev.actions)))
    assert all(ev.actions[k] == 0 and ev.changed[k] for k in order[:n_first])
    assert B.family_of("click on colour 9 (blue)") == "click" and B.family_of("ACTION3") == "ACTION3"
    assert np.array_equal(order, B.case_order(ev, "ACTION1")) and not np.array_equal(order, B.case_order(ev, "ACTION1", seed=1))


# ---- trust on every level, and the composed book (plan sections 4.4 and 4.5) -------------------
from gb.core import book as K  # noqa: E402
from gb.core.check import EXACT, UNKNOWN, WRONG  # noqa: E402

L1 = {"agent": 3, "step": 1}
# Half the slide: vouches for where the block arrives, not for the cells it leaves.
ARRIVES = SLIDE.replace('return out, (board == p["agent"]) | (out == p["agent"])', 'return out, out == p["agent"]')
# The other half: the cells the block leaves become background.
LEAVES = (SLIDE.replace('RULE = "ACTION1 slides', 'RULE = "ACTION1 empties the cells behind')
          .replace('return out, (board == p["agent"]) | (out == p["agent"])',
                   'return out, (board == p["agent"]) & (out != p["agent"])'))
# Claims the whole board and says ACTION1 does nothing.
STILL = "def applies(board, act, api): return act.action == 1\ndef predict(board, act, api): return board.copy()\n"
# Disagrees with SLIDE about the colour the block arrives in.
REPAINT = SLIDE.replace('out[ys, xs + p["step"]] = p["agent"]', 'out[ys, xs + p["step"]] = p["agent"] + 1') \
    .replace('return out, (board == p["agent"]) | (out == p["agent"])',
             'return out, np.roll(board == p["agent"], p["step"], axis=1)')


def member(src, binding=None, group="ACTION1", **kw):
    return dict(src=src, binding=binding or {}, group=group, **kw)


def test_trust_needs_every_level_and_pools_the_floors(tmp_path):
    ev1, p1 = level(tmp_path, "l1", LEVEL1)
    ev2, p2 = level(tmp_path, "l2", LEVEL2, seed=1, level=1)
    r1 = B.run_rule(SLIDE, p1, L1)
    r2 = B.run_rule(SLIDE, p2, {"agent": 9, "step": 2})
    one, both = K.trust([(r1, ev1)]), K.trust([(r1, ev1), (r2, ev2)])
    assert one["trusted"] and both["trusted"] and both["moves"] > one["moves"] and len(both["levels"]) == 2
    stale = B.run_rule(SLIDE, p2, {"agent": 9, "step": 1})       # level 2 with the wrong step
    t = K.trust([(r1, ev1), (stale, ev2)])
    assert not t["trusted"] and not t["exact_everywhere"] and t["levels"][0]["trusted_v2"]
    absent = B.run_rule(SLIDE, p2, L1)                           # colour 3 is not on level 2: nothing to be wrong on
    assert not absent["applies"].any() and K.trust([(r1, ev1), (absent, ev2)])["trusted"]
    tiny = SLIDE.replace('return out, (board == p["agent"]) | (out == p["agent"])',
                         'c = np.zeros((64, 64), bool); c[10, int(xs.max()) + 1] = True; return out, c')
    few, _ = level(tmp_path, "few", LEVEL1, n=30)
    narrow = K.trust([(B.run_rule(tiny, str(tmp_path / "few.npz"), L1), few)])
    assert not narrow["admitted"] and ("claim too narrow" in narrow["reason"] or "only" in narrow["reason"])
    ev, a1, a2 = conflicted(n_major=39, n_minor=1)               # one screen seen to end two ways, once
    p = str(tmp_path / "conf.npz")
    ev.save(p)
    paint = ("def applies(board, act, api): return act.action == 1 and board[0, 0] == 0\n"
             "def predict(board, act, api):\n    out = board.copy(); out[5, 5] = 3\n"
             "    c = np.zeros((64, 64), bool); c[5, 5] = True\n    return out, c\n")
    t = K.trust([(B.run_rule(paint, p), ev)])
    assert t["admitted"] and t["exact_everywhere"] and t["conflict_cases"] == 1
    assert not t["trusted"] and "more than one way" in t["reason"]


def test_two_partial_rules_compose_into_whole_boards(tmp_path):
    ev, path = level(tmp_path, "l1", LEVEL1)
    slide = ev.changed & (ev.actions == 0)
    click = ev.changed & (ev.actions >= 5)
    half = K.compose([member(ARRIVES, L1)], path)
    assert half["stage"] == "checked" and (half["status"][slide] == K.PARTIAL).all()
    assert (half["status"][click] == UNKNOWN).all() and (half["cells_right"][slide] == 2).all()
    n = K.numbers(half, ev)
    assert n["claimed_exactness"] == 1.0 and n["case_coverage"] == 0.0 and n["wrong"] == 0.0 and 0 < n["cell_coverage"] < 0.5
    assert not moves_right(ev, half["hash"])[slide[ev.moves["key"]]].any()   # half a move is not the next screen

    whole = K.compose([member(ARRIVES, L1), member(LEAVES, L1),
                       member(LIGHT, {"tile": 5, "lit": 6}, group="click on colour 5 (black)")], path)
    assert (whole["status"][ev.changed] == EXACT).all() and (whole["n_rules"][slide] == 2).all()
    n = K.numbers(whole, ev)
    assert (n["rules"], n["case_coverage"], n["cell_coverage"], n["wrong"], n["clash_cases"]) == (3, 1.0, 1.0, 0.0, 0)
    assert moves_right(ev, whole["hash"]).all()                  # unclaimed cells read as unchanged
    empty = K.compose([], path)
    assert (empty["status"] == UNKNOWN).all() and np.array_equal(empty["hash"], nothing_hash(ev))


def test_the_joining_rule_keeps_what_adds_cells_and_at_most_four_per_group(tmp_path):
    ev, path = level(tmp_path, "l1", LEVEL1, n=1500)
    picked = K.compose([member(ARRIVES, L1), member(ARRIVES + "# again\n", L1), member(LEAVES, L1)], path, select=True)
    assert picked["book"] == [0, 2] and "claim too narrow" in picked["notes"][1] and "0 cases" in picked["notes"][1]
    forced = K.compose([member(ARRIVES, L1), member(ARRIVES + "# again\n", L1, locked=True)], path, select=True)
    assert forced["book"] == [0, 1]
    per_tile = [member(LIGHT.replace("return act.action == 6 and", f"return act.click is not None and act.click[1] == {4 + 4 * i} and"),
                       {"tile": 5, "lit": 6}, group="click") for i in range(6)]
    full = K.compose(per_tile, path, select=True)
    assert full["book"] == [0, 1, 2, 3] and "already has 4 rules" in full["notes"][4]
    other = K.compose(per_tile[:4] + [dict(per_tile[4], group="another group")], path, select=True)
    assert other["book"] == [0, 1, 2, 3, 4]                       # the cap is per group


def test_a_cell_claimed_two_ways_is_unknown_and_both_rules_are_flagged(tmp_path):
    ev, path = level(tmp_path, "l1", LEVEL1)
    slide = ev.changed & (ev.actions == 0)
    res = K.compose([member(SLIDE, L1), member(REPAINT, L1)], path)
    # the four cells the block arrives in: one rule says the agent's colour, the other the next colour up
    assert (res["cells_clash"][slide] == 4).all() and res["clashes"][0] == res["clashes"][1] == 4 * slide.sum()
    assert (res["status"][slide] == K.PARTIAL).all()             # the cells it still claims are right; none is wrong
    assert (res["cells_right"][slide] == 2).all() and K.numbers(res, ev)["clash_cases"] == slide.sum()
    alone = K.compose([member(REPAINT, L1)], path)               # without the clash the same rule is simply wrong
    assert (alone["status"][slide] == WRONG).all() and K.numbers(alone, ev)["wrong"] > 0


def test_claiming_a_cell_stays_the_same_is_wrong_when_it_changed(tmp_path):
    ev, path = level(tmp_path, "l1", LEVEL1)
    slide = ev.changed & (ev.actions == 0)
    res = K.compose([member(STILL)], path)
    assert (res["status"][slide] == WRONG).all() and (res["cells_wrong"][slide] == 4).all()
    still_right = (ev.actions == 0) & ~ev.changed                # at the wall nothing moves: the rule is right there
    assert (res["status"][still_right] == EXACT).all()
    n = K.numbers(res, ev)
    assert n["wrong"] == round(slide.sum() / ev.changed.sum(), 4) and n["cell_coverage"] == 0.0
    crash = K.compose([member(SLIDE.replace('xs.max() + p["step"] < 64', "undefined_name"), L1), member(LEAVES, L1)], path)
    assert crash["book"] == [1] and "failed to run" in crash["notes"][0]
    assert K.compose([member("def applies(b, a, c): return True\n")], path)["stage"] == "static"


def test_the_book_on_the_next_level_uses_that_levels_bindings(tmp_path):
    ev2, p2 = level(tmp_path, "l2", LEVEL2, seed=1, level=1)
    rules = [member(SLIDE, L1), member(LIGHT, {"tile": 5, "lit": 6}, group="click")]
    t0 = K.compose(rules, p2)                                     # untouched: level 1's colours are not there
    assert (t0["status"] == UNKNOWN).all() and K.numbers(t0, ev2)["case_coverage"] == 0.0
    rebound = [member(SLIDE, {"agent": 9, "step": 2}), member(LIGHT, {"tile": 11, "lit": 12}, group="click")]
    t1 = K.compose(rebound, p2)
    n = K.numbers(t1, ev2)
    assert (n["case_coverage"], n["wrong"]) == (1.0, 0.0) and moves_right(ev2, t1["hash"]).all()


def test_what_the_book_leaves_unexplained(tmp_path):
    ev, path = level(tmp_path, "l1", LEVEL1)
    slide = ev.changed & (ev.actions == 0)
    half = K.compose([member(ARRIVES, L1)], path)
    left = K.unexplained(half, ev, "ACTION1")
    assert sorted(left) == np.flatnonzero(slide).tolist()
    k = next(iter(left))
    assert left[k].sum() == 2 and (ev.before[k][left[k]] == 3).all() and (ev.after[k][left[k]] == 0).all()
    every = K.unexplained(half, ev)                              # the clicks: nothing claims them at all
    clicks = [c for c in every if ev.actions[c] >= 5]
    assert clicks and all(np.array_equal(every[c], ev.before[c] != ev.after[c]) for c in clicks)
    assert K.unexplained(K.compose([member(SLIDE, L1)], path), ev, "ACTION1") == {}


def test_a_wrong_binding_that_applies_is_shown_beside_exact_ones_that_say_nothing(tmp_path):
    ev2, p2 = level(tmp_path, "l2", LEVEL2, seed=1, level=1)
    one_step = SLIDE.replace("Int(1, 3)", "Int(1, 1)")           # level 2 needs step 2: no binding of it is exact
    found = B.search(one_step, p2, family="ACTION1")
    assert found["exact"] > 0 and found["exact_applying"] == 0   # the colours that never move are "exact"
    assert B.choose(found) == (None, "binding failed")
    tried = [f for f in found["finals"] if not f["exact"]]
    assert tried and tried[0]["binding"] == {"agent": 9, "step": 1} and tried[0]["res"]["applies"].any()
    assert not tried[0]["res"]["claimed_ok"][tried[0]["res"]["applies"]].all()
