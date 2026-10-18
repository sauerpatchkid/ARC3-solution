"""H3 TEMPLATES v0 (custom_agents/gb/h3_templates.py): the smoke test's catalog and its guard."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import h3_templates as H  # noqa: E402


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
            H.try_game(run, game, [])
