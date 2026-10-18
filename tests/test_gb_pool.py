"""S2 POOL (custom_agents/gb/s2_pool.py): candidates of finished runs pooled, with no LLM."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import artifacts as A  # noqa: E402
from gb import s2_pool as P  # noqa: E402


def make_source(root, name, cands, groups, guarded=(), evidence=b"same"):
    d = root / name
    (d / "s2_write" / "toy").mkdir(parents=True)
    (d / "s3_check" / "toy").mkdir(parents=True)
    (d / "evidence").mkdir()
    (d / "evidence" / "toy_L1_train.npz").write_bytes(evidence)
    with open(d / "s2_write" / "toy" / "candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    (d / "s2_write" / "toy" / "groups.json").write_text(json.dumps(groups))
    if guarded:
        with open(d / "s3_check" / "toy" / "guarded.jsonl", "w") as f:
            for c in guarded:
                f.write(json.dumps(c) + "\n")
    return SimpleNamespace(name=name, dir=str(d))


def cand(i, code, group="ACTION1", binding=None, **kw):
    return dict(id=i, code=code, group=group, binding=binding or {}, order=i, stage="checked", lint=[], **kw)


def test_pool_renumbers_marks_duplicates_and_keeps_rescued_rules(tmp_path):
    a = make_source(tmp_path, "a", [cand(0, "x"), cand(1, "y", "click on colour 9 (blue)")],
                    ["ACTION1", "click on colour 9 (blue)"])
    b = make_source(tmp_path, "b", [cand(0, "x"), cand(1, "z", "ACTION2")], ["ACTION2", "ACTION1"],
                    guarded=[dict(cand("1.narrowed", "z2", "ACTION2"), kind="narrowed", trusted=True),
                             dict(cand("1.exception", "z3", "ACTION2"), kind="exception", trusted=False)])
    cands, groups, counts = P.pooled([a, b], "toy")
    assert [c["id"] for c in cands] == [0, 1, 2, 3, 4] == [c["order"] for c in cands]
    assert [c["pooled_from"] for c in cands] == [{"run": "a", "id": 0}, {"run": "a", "id": 1}, {"run": "b", "id": 0},
                                                  {"run": "b", "id": 1}, {"run": "b", "id": "1.narrowed"}]
    assert [c["duplicate"] for c in cands] == [False, False, True, False, False]
    assert groups == ["ACTION1", "click on colour 9 (blue)", "ACTION2"]
    assert counts == {"a": {"candidates": 2, "rescued": 0}, "b": {"candidates": 2, "rescued": 1}}


def test_pool_refuses_sources_with_different_evidence(tmp_path):
    a = make_source(tmp_path, "a", [cand(0, "x")], ["ACTION1"])
    b = make_source(tmp_path, "b", [cand(0, "y")], ["ACTION1"], evidence=b"other")
    run = A.Run("pooled", root=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="evidence differs"):
        P.run_game(run, "toy", [a, b], log=lambda m: None)


def test_pool_writes_s2s_files_and_copies_the_evidence(tmp_path):
    a = make_source(tmp_path, "a", [cand(0, "x")], ["ACTION1"])
    b = make_source(tmp_path, "b", [cand(0, "y")], ["ACTION1"])
    run = A.Run("pooled", root=str(tmp_path / "out"))
    P.run_game(run, "toy", [a, b], log=lambda m: None)
    s2 = os.path.join(run.dir, "s2_write", "toy")
    assert len(open(os.path.join(s2, "candidates.jsonl")).readlines()) == 2
    assert json.load(open(os.path.join(s2, "llm.json")))["llm_calls"] == 0
    assert open(os.path.join(run.dir, "evidence", "toy_L1_train.npz"), "rb").read() == b"same"
    assert P.run_game(run, "toy", [a, b], log=lambda m: None) is None        # current: skipped
