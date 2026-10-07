"""Rulebook v2's measuring tools (docs/plans/rulebook-v2.md section 3.3): claimed
cells, the fit/score split, T0, the tier lists and the artifact registry."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import benchmark  # noqa: E402
from test_wm_core import GOOD_MOVE, NOOP, OFF_BY_ONE, play  # noqa: E402
from test_wm_offline import write_corpus  # noqa: E402
from wm import registry, tiers  # noqa: E402
from wm.check import baseline_memory, book_status, check_rule  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.metrics import cell_coverage, changed_cells, claimed_grade, three_numbers, transfer_t0  # noqa: E402

# A v2 rule: it moves the block and vouches ONLY for the block's old and new cells.
PARTIAL = '''
RULE = "ACTION1 slides the colour-3 block one column right; only those cells are claimed"
def applies(board, act, api):
    return act.action == 1 and (board == 3).any() and np.nonzero(board == 3)[1].max() < 63
def predict(board, act, api):
    out = board.copy()
    ys, xs = np.nonzero(board == 3)
    out[board == 3] = api.background
    out[ys, xs + 1] = 3
    claimed = (board == 3) | (out == 3)
    return out, claimed
'''
# Right about the block, but it also scribbles on a cell it does NOT claim.
PARTIAL_MESSY = PARTIAL.replace("    claimed = (board == 3) | (out == 3)\n",
                                "    claimed = (board == 3) | (out == 3)\n    out[60, 60] = 9\n")
# Claims one cell only: exact, but explains almost nothing.
TINY_CLAIM = PARTIAL.replace("claimed = (board == 3) | (out == 3)",
                             "claimed = np.zeros((64, 64), bool); claimed[ys.min(), xs.max() + 1] = True")


@pytest.fixture(scope="module")
def ev(tmp_path_factory):
    f, n, a = play()
    e = LevelEvidence.from_moves("toy", 0, f, n, a)
    p = str(tmp_path_factory.mktemp("ev") / "toy.npz")
    e.save(p)
    return e, p


# ---- claimed cells ----------------------------------------------------------------
def test_a_v1_rule_claims_every_cell(ev):
    e, p = ev
    for src in (GOOD_MOVE, OFF_BY_ONE, NOOP):
        r = check_rule(src, p)
        ap = r["applies"]
        assert np.array_equal(r["claimed_ok"][ap], r["correct"][ap])
        assert (r["claimed_n"][ap] == int(e.live.sum())).all()
    g = claimed_grade(check_rule(GOOD_MOVE, p), e)
    assert g["trusted_v2"] and g["claimed_exactness"] == 1.0 and g["cells_false"] == 0


def test_a_partial_rule_is_graded_on_what_it_claims(ev):
    e, p = ev
    tidy, messy = check_rule(PARTIAL, p), check_rule(PARTIAL_MESSY, p)
    ap = tidy["applies"]
    assert tidy["claimed_ok"][ap].all() and (tidy["claimed_n"][ap] <= 8).all()
    assert claimed_grade(tidy, e)["trusted_v2"]
    # the scribble is outside the claim: wrong as a whole board, right on its claim
    assert not messy["correct"][ap].any() and messy["claimed_ok"][ap].all()
    assert claimed_grade(messy, e)["trusted_v2"]
    # a wrong rule is wrong on its claim too, and its false changes are counted
    bad = claimed_grade(check_rule(OFF_BY_ONE, p), e)
    assert not bad["admitted_v2"] and bad["cells_false"] > 0


def test_the_cell_floor_blocks_a_rule_that_claims_almost_nothing():
    b = play(1)[0][0]
    f = np.array([b] * 15)                     # the same move 15 times: one cell right each time
    n = f.copy()
    ys, xs = np.nonzero(b == 3)
    n[:, ys, xs] = 0
    n[:, ys, xs + 1] = 3
    e = LevelEvidence.from_moves("toy", 0, f, n, np.zeros(15, np.int32), mask=np.zeros((64, 64), bool))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        e.save(path)
        tiny = claimed_grade(check_rule(TINY_CLAIM, path), e)
    assert tiny["claimed_exact_all"] and tiny["cells_right"] == 15 and not tiny["admitted_v2"]


def test_the_three_numbers(ev):
    e, p = ev
    res = [check_rule(PARTIAL_MESSY, p)]
    st = book_status(res, len(e.actions))
    t = three_numbers(res, st, e)
    assert t["claimed_exactness"] == 1.0 and t["case_coverage"] == 0.0      # strict: every case "wrong"
    assert 0 < t["cell_coverage"] < 1                                       # clicks are not explained
    full = [check_rule(GOOD_MOVE, p)]
    assert cell_coverage(full, e) == t["cell_coverage"]                     # same cells explained
    assert int(changed_cells(e).sum()) > 0


# ---- fit / score split and T0 -------------------------------------------------------
def test_fit_and_score_do_not_share_moves(tmp_path):
    corpus = str(tmp_path / "run" / "transitions")
    write_corpus(corpus)
    train = LevelEvidence.from_corpus("toy", corpus, 0)
    fit, score = LevelEvidence.split("toy", corpus, 1, fit_moves=300, known_mask=train.mask)
    whole = LevelEvidence.from_corpus("toy", corpus, 1)
    assert int(fit.count.sum()) == 300 and int(fit.count.sum() + score.count.sum()) == int(whole.count.sum())
    assert np.array_equal(score.mask, whole.mask) and np.array_equal(score.first, whole.first)
    none_fit, none_score = LevelEvidence.split("toy", corpus, 7)
    assert none_fit is None and none_score is None
    short_fit, short_score = LevelEvidence.split("toy", corpus, 1, fit_moves=10 ** 6)
    assert short_fit is not None and short_score is None

    p_score = str(tmp_path / "score.npz")
    score.save(p_score)
    res = [check_rule(GOOD_MOVE, p_score)]
    t = transfer_t0(book_status(res, len(score.actions)), train, fit, score)
    assert t["book"] > t["nothing"] and t["wrong_on_changing"] == 0 and t["cases"] == len(score.actions)
    # memory given the fit split can only do as well or better than memory without it
    assert t["memory"] >= baseline_memory(train, score).mean() - 1e-9


# ---- tiers ----------------------------------------------------------------------------
def test_every_public_game_is_in_exactly_one_tier():
    listed = [g for games in tiers.TIERS.values() for g in games]
    assert sorted(listed) == sorted(benchmark.ALL_GAMES) and len(set(listed)) == 25
    assert tiers.tier_of("ft09") == "dev" and tiers.tier_of("wa30") == "untouched"
    assert set(tiers.BUILD_FROM) == set(tiers.DEV) | set(tiers.SEEN)


def test_the_recorded_held_out_draw_reproduces():
    ab, untouched = tiers.draw()
    assert ab == tuple(sorted(tiers.AB_OFFLINE)) and untouched == tuple(sorted(tiers.UNTOUCHED))
    assert len(set(ab) & set(tiers.NO_ACTION7)) == 3 == len(set(untouched) & set(tiers.NO_ACTION7))


# ---- registry -------------------------------------------------------------------------
def test_registry_freezes_and_notices_a_change(tmp_path):
    reg = str(tmp_path / "artifacts" / "registry.json")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "a.py").write_text("x = 1\n")
    (tmp_path / "lib" / "b.py").write_text("y = 2\n")
    e = registry.add("helpers_v1", ["lib"], ["ft09", "tu93"], path=reg, root=str(tmp_path), today="2026-10-28")
    assert e["n_files"] == 2 and e["source_games"] == ["ft09", "tu93"]
    assert registry.verify(path=reg, root=str(tmp_path)) == []
    with pytest.raises(SystemExit):                              # a name is used once
        registry.add("helpers_v1", ["lib"], ["ft09"], path=reg, root=str(tmp_path))
    (tmp_path / "lib" / "a.py").write_text("x = 2\n")
    os.remove(tmp_path / "lib" / "b.py")
    bad = registry.verify(path=reg, root=str(tmp_path))
    assert sorted(b[2] for b in bad) == ["changed", "missing"]
    registry.log_event("unsealed", "confirm 1 per-game results", path=reg, today="2026-10-28")
    assert registry.load(reg)["events"][0]["what"] == "unsealed"
