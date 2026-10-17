"""S3's guard fitter (custom_agents/gb/s3_guards.py) on synthetic games only: CP1's
candidates are never touched here, because the fitter's first run on them IS G0
(docs/3-glassbox/glassbox-prereg.md section 3).

  - the search, on hand-made feature bits: the simplest guard wins, a negation
    and a pair when needed, a guard that fits by chance fails the hold-out
  - the wrapper: the original code kept, the names it would clash with renamed,
    v1 rules and rules without PARAMS
  - a click game: the guard is a colour, becomes a PARAMS slot, and re-binds on
    a level where another colour plays that part
  - a maze: the whole stage on a run folder, from S1 to the book and T1 before
    and after, and its refusal to run on CP1's candidates
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from gb import artifacts as A  # noqa: E402
from gb import s1_see  # noqa: E402
from gb import s3_guards as G  # noqa: E402
from gb.core import bind as B  # noqa: E402
from gb.core import book as K  # noqa: E402
from gb.core import contract as C  # noqa: E402
from gb.core.evidence import LevelEvidence  # noqa: E402
from gb.core.lint import lint  # noqa: E402
from gb.core.metrics import memory_hash, moves_right, nothing_hash  # noqa: E402
from gb.cp1_import import member, score_book  # noqa: E402
from test_gb_see import BG, board  # noqa: E402

ZERO = np.zeros((64, 64), bool)


# ---- the search, on hand-made bits ---------------------------------------------------------------------
def fit(names, rows, wrong, **kw):
    bits = np.array(rows, bool)
    right = [k for k in range(len(bits)) if k not in wrong]
    return G.fit_guard(bits, names, right, wrong, dict(G.DEFAULTS, **kw))


def test_the_hold_out_is_seeded_and_keeps_wrong_cases():
    a = G.split(range(100), range(100, 110), 0.30, 0)
    assert a == G.split(range(100), range(100, 110), 0.30, 0)
    fr, fw, hr, hw = a
    assert (len(fr), len(fw), len(hr), len(hw)) == (70, 7, 30, 3)
    assert set(fw) | set(hw) == set(range(100, 110)) and not set(fw) & set(hw)
    assert len(G.split(range(10), range(3), 0.30, 0)[3]) == 1          # 3 wrong cases: 1 held out
    assert G.split(range(100), range(100, 110), 0.30, 1) != a


def test_one_geometric_fact_beats_a_colour_and_a_pair():
    names = ["subject_colour=4", "up_edge", "down_edge", "left_touch"]
    rows = [[1, 1, 1, 1]] * 6 + [[0, 1, 0, 0], [0, 0, 1, 0]] * 10       # cases 0-5 wrong
    g = fit(names, rows, list(range(6)))
    assert g["guard"] == [["left_touch", True]] and g["failed_holdout"] == 0


def test_a_negation_and_a_pair_when_nothing_simpler_fits():
    names = ["left_clear", "up_touch"]
    rows = [[0, 1]] * 5 + [[1, 1], [1, 0]] * 10
    assert fit(names, rows, list(range(5)))["guard"] == [["left_clear", False]]
    names = ["left_touch", "up_touch"]
    rows = [[1, 1]] * 5 + [[1, 0], [0, 1], [0, 0]] * 10                 # only both together
    assert fit(names, rows, list(range(5)))["guard"] == [["left_touch", True], ["up_touch", True]]
    assert fit(names, rows, list(range(5)), guard_max_literals=1)["guard"] is None


def test_a_guard_that_fits_by_chance_fails_the_hold_out():
    right, wrong = list(range(4, 40)), [0, 1, 2, 3]
    fr, fw, hr, hw = G.split(right, wrong, 0.30, 0)
    names = ["left_touch", "up_touch"]
    rows = np.zeros((40, 2), bool)
    rows[fw, 0] = True                     # left_touch: every fit wrong case, but not the held-out one
    rows[wrong, 1] = True                  # up_touch: every wrong case
    g = fit(names, rows, wrong)
    assert g["guard"] == [["up_touch", True]] and g["failed_holdout"] == 1
    rows[:, 1] = False
    g = fit(names, rows, wrong)
    assert g["guard"] is None and "fail on the held-out cases" in g["why"]


def test_too_few_wrong_cases_get_no_guard():
    g = fit(["left_touch"], [[1], [1], [0], [0], [0]], [0, 1])
    assert g["guard"] is None and g["why"].startswith("only 2 wrong cases")


def test_colours_can_be_switched_off():
    names = ["subject_colour=4", "left_touch"]
    rows = [[1, 0]] * 5 + [[0, 0], [0, 1]] * 10
    assert fit(names, rows, list(range(5)))["guard"] == [["subject_colour=4", True]]
    assert fit(names, rows, list(range(5)), guard_colour_features="off")["guard"] is None


# ---- the wrapper ----------------------------------------------------------------------------------------
V2 = '''# a v2 rule with a helper, a bind() and comments
RULE = "ACTION3 moves the agent left"
PARAMS = {
    "agent": Colour(),   # the player
}

def cells(board, p):
    return board == p["agent"]

def bind(board0, api):
    return {"agent": 9}

def applies(board, act, api, p):
    return act.action == 3 and bool(cells(board, p).any())

def predict(board, act, api, p):
    m = cells(board, p)
    out = np.array(board)
    ys, xs = np.nonzero(m)
    out[m] = api.background
    out[ys, xs - 1] = p["agent"]
    return out
'''


def test_the_wrapper_keeps_the_original_and_adds_the_guard():
    slots = {G.MOVER_SLOT: G.MOVER_SLOT, "left_colour=4": "guard_colour1"}
    src = G.wrap(V2, "narrowed", [("left_touch", True), ("left_colour=4", True)], slots,
                 {G.MOVER_SLOT: 9, "guard_colour1": 4}, "moves left unless", "words", 7)
    ok, why, version, params = C.check_form(src)
    assert ok, why
    assert list(params) == ["agent", G.MOVER_SLOT, "guard_colour1"]
    assert "# the player" in src and "def cells(board, p):" in src and "def base_bind(board0, api):" in src
    assert 'f"left_colour={p[\'guard_colour1\']}" in f' in src
    rule = C.Rule(src)
    assert rule.text == "moves left unless"
    assert rule.guess(LevelEvidence.from_moves("x", 0, board()[None], board()[None], np.zeros(1, np.int32),
                                               mask=ZERO).api)[0] == {"agent": 9, G.MOVER_SLOT: 9, "guard_colour1": 4}


def test_the_wrapper_renames_what_would_clash_and_takes_v1_rules():
    own = V2.replace("def cells(board, p):", "def features(board, p):").replace("cells(board, p)", "features(board, p)")
    src = G.wrap(own, "exception", [("left_touch", True)], {G.MOVER_SLOT: "agent"}, {}, "x", "w", 1)
    assert C.check_form(src)[0] and "def base_features(board, p):" in src and "base_features(board, p).any()" in src
    assert 'f = features(board, act, api, p["agent"])' in src
    v1 = "def applies(board, act, api):\n    return act.action == 1\n\ndef predict(board, act, api):\n    return board\n"
    src = G.wrap(v1, "narrowed", [("up_touch", True)], {G.MOVER_SLOT: G.MOVER_SLOT}, {G.MOVER_SLOT: 9}, "x", "w", 2)
    ok, why, version, params = C.check_form(src)
    assert ok and version == 2 and list(params) == [G.MOVER_SLOT] and "base_applies(board, act, api)" in src
    empty = V2.replace('PARAMS = {\n    "agent": Colour(),   # the player\n}', "PARAMS = {}").replace('p["agent"]', "9")
    src = G.wrap(empty, "narrowed", [("action3", True)], {}, {}, "x", "w", 3)
    assert C.check_form(src)[0] and "PARAMS = {}" in src
    with pytest.raises(G.GuardError):
        G.wrap(V2.replace("def cells", "def base_applies"), "narrowed", [("action3", True)], {}, {}, "x", "w", 4)


# ---- a click game: the guard is a colour -------------------------------------------------------------------
DOTS_RULE = '''
RULE = "Clicking a dot paints it"
PARAMS = {"paint": Colour()}

def applies(board, act, api, p):
    return act.click is not None and int(board[act.click[0], act.click[1]]) != api.background

def predict(board, act, api, p):
    y, x = act.click
    out = np.array(board)
    out[y, x] = p["paint"]
    claimed = np.zeros(board.shape, bool)
    claimed[y, x] = True
    return out, claimed
'''


def dots(seed, inert, others, n=600):
    """Dots on a lattice; a click paints a dot blue (9), except a dot of the `inert` colour."""
    rng = np.random.default_rng(seed)
    b = board()
    cells = [(y, x) for y in range(6, 60, 6) for x in range(6, 60, 6)]
    for y, x in cells:
        b[y, x] = rng.choice([inert, *others])
    F, N, acts = [], [], []
    for _ in range(n):
        y, x = cells[rng.integers(len(cells))]
        nb = b.copy()
        if b[y, x] != inert:
            nb[y, x] = 9
        F.append(b)
        N.append(nb)
        acts.append(5 + y * 64 + x)
        b = nb
    return LevelEvidence.from_moves("dots", seed, np.array(F), np.array(N), np.array(acts, np.int32), mask=ZERO)


def test_a_colour_guard_becomes_a_slot_and_rebinds_on_the_next_level(tmp_path):
    ev = dots(0, inert=4, others=(3, 12))
    p = str(tmp_path / "dots_L0.npz")
    ev.save(p)
    res = B.run_rule(DOTS_RULE, p, binding={"paint": 9})
    arr, _, role, _ = s1_see.see_level(ev)
    names = [str(x) for x in arr["feature_names"]]
    bits = np.unpackbits(arr["features"], axis=1)[:, :len(names)].astype(bool)
    c = {"id": 3, "group": "click on colour 4 (darker gray)", "code": DOTS_RULE, "binding": {"paint": 9}, "order": 0}
    rec, made = G.fit_rule(c, res["applies"], res["claimed_ok"], ev.actions, bits, names, role["mover"], G.DEFAULTS)
    assert rec["guard"] == [["subject_colour=4", True]] and rec["words"] == "the clicked object is colour 4"
    narrowed, exception = made
    assert narrowed["binding"] == {"paint": 9, "guard_colour1": 4} and G.MOVER_SLOT not in narrowed["binding"]
    assert "the clicked object is colour <guard_colour1>" in C.Rule(narrowed["code"]).text
    for m in made:                                  # both exact on their level; the exception is a no-op
        _, t = G.replay(m, p, ev)
        assert t["exact_everywhere"] and t["moves"] >= 20
        assert t["trusted"] if m is narrowed else G.noop_exact(t)
    nxt = dots(1, inert=3, others=(4, 12))          # now 3 is the colour that does nothing
    p2 = str(tmp_path / "dots_L1.npz")
    nxt.save(p2)
    best, flag = B.choose(B.search(narrowed["code"], p2, family="click", prefer=narrowed["binding"]))
    assert best["binding"] == {"paint": 9, "guard_colour1": 3} and flag is None


# ---- a maze: the whole stage on a run folder ---------------------------------------------------------------
STEP = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def maze(seed, mover, n):
    """A 2x2 mover in a grey frame with grey 2x2 pillars. ACTION1-4 move it a cell
    up, down, left, right, unless something is in the way. The pillars are spaced
    so there is no dead end (test_gb_vocab's known limit); in a corner the mover
    touches as much wall as floor, and sits on the floor."""
    rng = np.random.default_rng(seed)
    b = board()
    b[14, 14:51] = b[50, 14:51] = b[14:51, 14] = b[14:51, 50] = 4
    spots = [(y, x) for y in range(19, 45, 7) for x in range(19, 45, 7) if abs(y - 32) > 3 or abs(x - 32) > 3]
    for i in rng.choice(len(spots), 8, replace=False):
        y, x = spots[i]
        b[y:y + 2, x:x + 2] = 4
    b[32:34, 32:34] = mover
    F, N, acts = [], [], []
    for _ in range(n):
        a = int(rng.integers(0, 4))
        dy, dx = STEP[a]
        own = b == mover
        ys, xs = np.nonzero(own)
        nb = b.copy()
        if (b[ys + dy, xs + dx][~own[ys + dy, xs + dx]] == BG).all():
            nb[own] = BG
            nb[ys + dy, xs + dx] = mover
        F.append(b)
        N.append(nb)
        acts.append(a)
        b = nb
    order = {"num": np.arange(n), "attempt": np.zeros(n, np.int32), "t": np.arange(n, dtype=np.int32)}
    return np.array(F), np.array(N), np.array(acts, np.int32), order


def move_rule(action, dy, dx, blocked_check):
    head = f'''
RULE = "ACTION{action} moves the agent one cell"
PARAMS = {{"agent": Colour()}}

def applies(board, act, api, p):
    if act.action != {action}:
        return False
    m = board == p["agent"]
    ys, xs = np.nonzero(m)
    if not len(ys):
        return False
'''
    check = f'''    ty, tx = ys + ({dy}), xs + ({dx})
    if ty.min() < 0 or tx.min() < 0 or ty.max() > 63 or tx.max() > 63:
        return False
    return bool(((board[ty, tx] == api.background) | m[ty, tx]).all())
''' if blocked_check else "    return True\n"
    return head + check + f'''
def predict(board, act, api, p):
    m = board == p["agent"]
    ys, xs = np.nonzero(m)
    out = np.array(board)
    out[m] = api.background
    out[ys + ({dy}), xs + ({dx})] = p["agent"]
    claimed = m.copy()
    claimed[ys + ({dy}), xs + ({dx})] = True
    return out, claimed
'''


RULES = {7: ("ACTION3", move_rule(3, 0, -1, False)),           # near misses: wrong where blocked
         8: ("ACTION4", move_rule(4, 0, 1, False)),
         9: ("ACTION1", move_rule(1, -1, 0, True))}            # exact: it checks for itself


def quiet(_):
    pass


def make_run(root, source="test", record=True):
    """A run folder in the shape cp1_import.py writes: evidence (a training level,
    and the next level's fit and score splits, where the mover is another colour),
    S1's facts, candidates, their grades and per-case results, and the book's
    group order."""
    run = A.Run("synthetic", root=str(root))
    ev_dir = os.path.join(run.dir, "evidence")
    os.makedirs(ev_dir, exist_ok=True)
    f, n, a, o = maze(0, 9, 2000)
    train = LevelEvidence.from_moves("maze", 0, f, n, a, mask=ZERO, order=o)
    p_train = os.path.join(ev_dir, "maze_L0_train.npz")
    train.save(p_train)
    f, n, a, o = maze(1, 12, 1500)
    part = lambda s: {k: v[s] for k, v in o.items()}
    fit = LevelEvidence.from_moves("maze", 1, f[:300], n[:300], a[:300], mask=ZERO, order=part(slice(None, 300)))
    score = LevelEvidence.from_moves("maze", 1, f[300:], n[300:], a[300:], mask=ZERO, order=part(slice(300, None)))
    p_fit, p_score = os.path.join(ev_dir, "maze_L1_fit.npz"), os.path.join(ev_dir, "maze_L1_score.npz")
    fit.save(p_fit)
    score.save(p_score)
    s1_see.run_game(run, "maze", log=quiet)
    cands, rules, per_case = [], [], []
    for order, (rid, (group, code)) in enumerate(RULES.items()):
        res = B.run_rule(code, p_train, binding={"agent": 9})
        t = K.trust([(res, train)])
        c = {"id": rid, "group": group, "code": code, "binding": {"agent": 9}, "order": order, "lint": lint(code),
             "origin": "first round", "duplicate": False, "stage": "checked", "trust": t, "source": source}
        if t["trusted"]:
            c["t1"] = G.carry(c, res, train, fit, p_fit)
        cands.append(c)
        rules.append({"id": rid, "group": group, "v2_admitted": t["admitted"], "v2_trusted": t["trusted"], "trust": t})
        per_case.append(res)
    s2, s3 = os.path.join(run.stage_dir("s2_write"), "maze"), os.path.join(run.stage_dir("s3_check"), "maze")
    os.makedirs(s2, exist_ok=True)
    os.makedirs(s3, exist_ok=True)
    for path, rows in ((os.path.join(s2, "candidates.jsonl"), cands), (os.path.join(s3, "rules.jsonl"), rules)):
        with open(path, "w") as fh:
            fh.writelines(json.dumps(G.plain(r)) + "\n" for r in rows)
    np.savez_compressed(os.path.join(s3, "rule_cases.npz"), ids=np.array(list(RULES)),
                        applies=np.array([r["applies"] for r in per_case]),
                        claimed_ok=np.array([r["claimed_ok"] for r in per_case]))
    book = {"game": "maze", "level": 0, "groups_called": ["ACTION3", "ACTION4", "ACTION1"]}
    if record:                     # the book and T1 as cp1_import records them (only rule 9 is trusted)
        trusted = [c for c in cands if c["trust"]["trusted"]]
        book["numbers"] = K.numbers(K.compose([member(c, p_train) for c in trusted], p_train, select=True), train)
        base = {"nothing": moves_right(score, nothing_hash(score)),
                "memory": moves_right(score, memory_hash([train, fit], score))}
        t1, _ = score_book([member(c, p_fit, c["t1"]["binding"]) for c in trusted if c["t1"]["in_t1"]],
                           p_score, score, base)
        os.makedirs(run.stage_dir("s4_carry"), exist_ok=True)
        with open(os.path.join(run.stage_dir("s4_carry"), "maze_L1.carry.json"), "w") as fh:
            json.dump(G.plain({"t1": t1}), fh)
    with open(os.path.join(run.stage_dir("s3_check"), "maze_L0.book.json"), "w") as fh:
        json.dump(G.plain(book), fh)
    return run, rules


@pytest.fixture(scope="module")
def maze_run(tmp_path_factory):
    run, rules = make_run(tmp_path_factory.mktemp("glassbox"))
    return run, rules, G.run_game(run, "maze", log=quiet)


def test_the_maze_has_two_near_misses_and_one_exact_rule(maze_run):
    _, rules, _ = maze_run
    assert [(r["v2_admitted"], r["v2_trusted"]) for r in rules] == [(True, False), (True, False), (True, True)]


def test_the_blocked_moves_are_guarded_and_the_narrowed_rules_trusted(maze_run):
    _, _, s = maze_run
    assert (s["looked_at"], s["guards"]) == (2, 2)
    got = {r["parent"]: r for r in s["rescued"]}
    assert set(got) == {7, 8}
    assert got[7]["guard"] == "the mover is touching something on its left"
    assert got[8]["guard"] == "the mover is touching something on its right"
    assert all(r["exactness_before"] < 1 and r["exactness_after"] == 1 and r["in_book"] for r in got.values())


def test_the_book_covers_more_after(maze_run):
    _, _, s = maze_run
    before, after = s["book"]["before"], s["book"]["after"]
    assert before["members"] == [9] and set(after["members"]) == {9, "7.narrowed", "8.narrowed"}
    assert after["cell_coverage"] > before["cell_coverage"] and after["wrong"] == 0


def test_t1_rebinds_the_narrowed_rules_to_the_new_mover(maze_run):
    run, _, s = maze_run
    assert s["t1"]["before"]["members"] == [9]
    assert set(s["t1"]["after"]["members"]) == {9, "7.narrowed", "8.narrowed"}
    assert s["t1"]["after"]["per_move"]["rate"]["book"] > s["t1"]["before"]["per_move"]["rate"]["book"]
    with open(os.path.join(run.dir, "s3_check", "maze", "guarded.jsonl")) as fh:
        new = {c["id"]: c for c in map(json.loads, fh)}
    assert new["7.narrowed"]["binding"] == {"agent": 9}                  # the guard uses the rule's own slot
    assert 'features(board, act, api, p["agent"])' in new["7.narrowed"]["code"]
    assert new["7.narrowed"]["t1"]["binding"] == {"agent": 12}
    assert set(new) == {"7.narrowed", "7.exception", "8.narrowed", "8.exception"}


def test_before_is_checked_against_what_the_import_recorded(maze_run, tmp_path):
    _, _, s = maze_run
    assert s["before_matches_import"] == {"book": True, "t1": True}
    run, _ = make_run(tmp_path, record=False)
    assert G.run_game(run, "maze", log=quiet)["before_matches_import"] == {"book": None, "t1": None}


def test_the_stage_is_skipped_when_current(maze_run):
    run, _, _ = maze_run
    assert G.run_game(run, "maze", log=quiet) is None
    h = run.read_header("s3_check", "maze.guards")
    assert set(h["outputs"]) == {"guards", "guarded", "summary"} and h["version"] == G.VERSION


def test_cp1s_candidates_are_refused_without_g0(tmp_path):
    run = A.Run("cp1_like", root=str(tmp_path))
    os.makedirs(os.path.join(run.stage_dir("s2_write"), "maze"))
    with open(os.path.join(run.stage_dir("s2_write"), "maze", "candidates.jsonl"), "w") as fh:
        fh.write(json.dumps({"id": 0, "source": "cp1"}) + "\n")
    with pytest.raises(RuntimeError, match="IS G0"):
        G.run_game(run, "maze", log=quiet)
