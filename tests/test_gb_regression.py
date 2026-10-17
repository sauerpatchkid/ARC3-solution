"""The CP1 regression check (plan section 2.5, rule 2): gb/core, Glassbox's copy of
Rulebook v2's code, must reproduce CP1's registered numbers exactly.

Re-runs four of CP1's games with gb/core (no LLM, no search; about 15 s) and
compares every recomputed number with the registered one. Together they cover a
book with several rules and a full transfer (dc22, ft09), a two-level repair
(vc33) and an empty book (tu93). The whole set of 8 games:

    uv run python tools/gb_offline.py --import-cp1      # ~80 s, 437 numbers

Run this before every commit that touches custom_agents/gb/core/. It needs
CP1's registered results (results/ is not in git), so it is skipped without them.
"""
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

import registry  # noqa: E402
from gb import artifacts as A  # noqa: E402
from gb import cp1_import  # noqa: E402

GAMES = ("tu93", "vc33", "ft09", "dc22")
pytestmark = pytest.mark.skipif(not os.path.isdir(cp1_import.CP1_DIR),
                                reason="CP1's registered results are not on this machine")


@pytest.fixture(scope="module")
def checks(tmp_path_factory):
    run = A.Run("cp1_regression", root=str(tmp_path_factory.mktemp("glassbox")))
    return {g: cp1_import.import_game(g, run, log=lambda m: None) for g in GAMES}, run


@pytest.mark.parametrize("game", GAMES)
def test_gb_core_reproduces_cp1(checks, game):
    check = checks[0][game]
    bad = [i["what"] for i in check.items if not i["match"]]
    assert check.items and not bad, f"{game}: gb/core no longer reproduces CP1 on {bad[:5]}"


def test_the_import_writes_glassbox_artifacts_and_never_touches_cp1(checks):
    _, run = checks
    h = run.require("s3_check", "dc22", cp1_import.VERSION)
    assert set(h["outputs"]) == {"rules", "rule_cases", "book", "status"}
    assert run.require("s4_carry", "ft09", cp1_import.VERSION)["inputs"]["cp1_result"]["sha256"]
    assert registry.verify("rulebook_v2_cp1_tier0a") == []          # CP1's registered files are unchanged


def test_the_stages_with_s0_off_give_cp1s_book_and_transfer(tmp_path):
    # S0 (bars off) -> S3 -> S4 on dc22, whose book has 3 rules that all carry: CP1's book, T0, T1 and
    # re-bindings, the last found again by S4's own search (make gb-g0b-control checks all 8 games).
    from gb import s0_mask, s3_check, s4_carry
    src = A.Run("cp1_src", root=str(tmp_path))
    cp1_import.import_game("dc22", src, log=lambda m: None)
    run = A.Run("control", root=str(tmp_path))
    run.config.setdefault("mask", {})["complete_bars"] = False
    res = cp1_import.load_cp1("dc22")
    os.makedirs(os.path.join(run.stage_dir("s2_write"), "dc22"))
    for f in ("candidates.jsonl", "llm.json", "groups.json"):
        shutil.copyfile(os.path.join(src.dir, "s2_write", "dc22", f), os.path.join(run.dir, "s2_write", "dc22", f))
    s0_mask.run_game(run, "dc22", res["corpus"], res["train"]["level"], res["transfer_level"], log=lambda m: None)
    s3_check.run_game(run, "dc22", log=lambda m: None)
    s4_carry.run_game(run, "dc22", log=lambda m: None)
    got = cp1_import.matches(run, "dc22")
    assert set(got) == {"members", "book numbers", "t0", "t1", "re-bindings"} and all(got.values()), got
