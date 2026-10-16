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
    assert set(h["outputs"]) == {"rules", "book", "status"}
    assert run.require("s4_carry", "ft09", cp1_import.VERSION)["inputs"]["cp1_result"]["sha256"]
    assert registry.verify("rulebook_v2_cp1_tier0a") == []          # CP1's registered files are unchanged
