"""H3 TEMPLATES (custom_agents/gb/h3_templates.py): version 1's catalog, its guards and its
use by S2; version 0's smoke catalog and guard."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from gb import h3_templates as H  # noqa: E402
from test_gb_guards import move_rule  # noqa: E402

RULE = move_rule(1, -1, 0, True)
BIND_CLEAN = RULE + '''
def bind(board0, api):
    return {"agent": int(board0.max())}
'''
BIND_FACT = move_rule(4, 0, 1, True) + '''
def bind(board0, api):
    return {"agent": int(board0[4, 4])}
'''


def trusted(i, code, group="ACTION1", **kw):
    return dict(id=i, code=code, group=group, binding={"agent": 9}, lint=[], stage="checked", rule=f"rule {i}",
                trust={"trusted": True, "cells_right": 100 - i, "moves": 50}, **kw)


def fake_book(tmp_path, game, rows):
    d = tmp_path / "book"
    (d / "s3_check").mkdir(parents=True, exist_ok=True)
    (d / "s2_write" / game).mkdir(parents=True, exist_ok=True)
    (d / "s3_check" / f"{game}_L1.book.json").write_text(json.dumps({"level": 1, "members": [r["id"] for r in rows]}))
    with open(d / "s2_write" / game / "candidates.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(d)


# ---- version 1 ------------------------------------------------------------------------------------
def test_bind_is_kept_only_without_level_facts():
    assert H.checked_bind(BIND_CLEAN) == (BIND_CLEAN, None)
    code, why = H.checked_bind(BIND_FACT)
    assert "def bind" not in code and "def applies" in code and "coordinate" in why
    assert H.checked_bind(RULE) == (RULE, None)


def test_kind_is_read_from_the_code():
    assert H.rule_kind('KIND = "move"\n' + RULE) == "move"
    assert H.rule_kind(RULE) is None


def test_the_catalog_keeps_trusted_clean_rules_with_params_once(tmp_path):
    rows = [trusted(0, RULE), trusted(1, RULE),                          # the same code twice: one template
            trusted(2, BIND_FACT, group="ACTION4"),                      # kept, without its bind()
            dict(trusted(3, move_rule(3, 0, -1, True)), lint=[{"kind": "colour literal"}]),
            dict(trusted(4, move_rule(2, 1, 0, True)), binding={}),     # no PARAMS bound: cannot carry
            dict(trusted(5, move_rule(3, 0, -1, False)), trust={"trusted": False, "cells_right": 1, "moves": 50}),
            trusted(6, BIND_CLEAN)]          # RULE with a bind(): the same answers on 59 bank moves, one template
    catalog, report = H.build_catalog_v1([fake_book(tmp_path, "tu93", rows)], log=lambda m: None)
    assert sorted(t["key"] for t in catalog) == ["tu93/0", "tu93/2"]
    assert [d["dropped"] for d in report["duplicate_code"]] == ["tu93/1"]
    assert report["duplicate_behaviour"] == [{"dropped": "tu93/6", "kept": "tu93/0"}]
    assert [r["candidate"] for r in report["bind_removed"]] == [2]
    t = next(t for t in catalog if t["key"] == "tu93/2")
    assert not t["bind_kept"] and "def bind" not in t["code"] and t["family"] == "button" and t["trusted_on"] == [1]


def test_the_catalog_refuses_a_held_out_game(tmp_path):
    with pytest.raises(ValueError, match="dev and seen games only"):
        H.build_catalog_v1([fake_book(tmp_path, "ar25", [trusted(0, RULE)])], log=lambda m: None)


def test_try_game_never_touches_the_untouched_tier_and_needs_g3_for_the_ab_tier(tmp_path):
    run = SimpleNamespace(name="x", dir=str(tmp_path))
    with pytest.raises(ValueError, match="untouched"):
        H.try_game(run, "bp35", [], [], g3=True)
    with pytest.raises(ValueError, match="only in G3"):
        H.try_game(run, "ar25", [], [])


def test_an_unregistered_catalog_is_refused(tmp_path):
    p = tmp_path / "catalog.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit, match="not registered"):
        H.load_catalog(str(p))


def test_s2_starts_a_group_from_its_template_and_skips_the_first_round(tmp_path):
    from gb import artifacts as A, s1_see
    from gb import s2_write as W
    from gb.core.evidence import LevelEvidence
    from gb.core.llm import ChatClient
    from test_gb_guards import ZERO, maze
    run = A.Run("t", root=str(tmp_path))
    os.makedirs(os.path.join(run.dir, "evidence"))
    f, n, a, o = maze(0, 9, 2000)
    LevelEvidence.from_moves("maze", 0, f, n, a, mask=ZERO, order=o).save(
        os.path.join(run.dir, "evidence", "maze_L0_train.npz"))
    s1_see.run_game(run, "maze", log=lambda m: None)
    asked = []

    def refuse(messages, n):
        asked.append(n)
        return ["no code"] * n
    client = ChatClient(mock=refuse, cache_path=os.path.join(run.stage_dir("s2_write"), "llm_cache.jsonl"))
    p_train, ev, texts, bits, names, mover = W.level_inputs(run, "maze")
    templates = {"ACTION1": [{"tid": 7, "code": RULE, "binding": {"agent": 9}}]}
    w = W.Writer("maze", ev, p_train, client, W.DEFAULTS, texts, bits, names, mover, True, lambda m: None, templates)
    rec = w.run_group("ACTION1")
    assert rec["first_round"] == 0 and rec["templates"] == 1 and not asked
    assert rec["arms"][0]["origin"] == "template" and rec["arms"][0]["template"] == 7 and W.carries(rec["arms"][0])
    plain = W.Writer("maze", ev, p_train, client, W.DEFAULTS, texts, bits, names, mover, True, lambda m: None)
    assert plain.run_group("ACTION1")["first_round"] == 1 and asked[0] == W.DEFAULTS["k_first"]


# ---- version 0 (the smoke test) -------------------------------------------------------------------
def test_family_is_click_or_button():
    assert H.family("click on colour 9 (blue)") == "click"
    assert H.family("ACTION3") == "button"


def test_catalog_keeps_book_members_with_params_and_no_lint_once_per_code(tmp_path):
    d = tmp_path / "book"
    (d / "s3_check").mkdir(parents=True)
    (d / "s2_write" / "ft09").mkdir(parents=True)
    (d / "s3_check" / "ft09_L1.book.json").write_text(json.dumps({"level": 1, "members": [0, 1, 2, 3]}))
    rows = [dict(id=0, code="a", group="ACTION1", binding={"x": 1}, lint=[]),
            dict(id=1, code="b", group="ACTION1", binding={}, lint=[]),            # no PARAMS: cannot carry
            dict(id=2, code="c", group="click on colour 9 (blue)", binding={"x": 1}, lint=["level fact"]),
            dict(id=3, code="a", group="ACTION2", binding={"x": 2}, lint=[]),      # same code as 0
            dict(id=4, code="d", group="ACTION1", binding={"x": 1}, lint=[])]      # not in the book
    with open(d / "s2_write" / "ft09" / "candidates.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cat = H.build_catalog(str(d), ["ft09", "tu93"])
    assert [(t["source_id"], t["family"]) for t in cat] == [(0, "button")]


def test_the_smoke_test_never_touches_a_held_out_game(tmp_path):
    run = SimpleNamespace(name="x", dir=str(tmp_path))
    for game in ("ar25", "bp35", "ft09"):
        with pytest.raises(ValueError, match="not a seen game"):
            H.smoke_game(run, game, [])
