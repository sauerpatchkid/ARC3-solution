"""The adopted Goose as a named agent (custom_agents/presets.py)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

import run_local  # noqa: E402
run_local._install_minimal_agents_pkg()          # provides agents.structs
import presets  # noqa: E402
from custom_agents import REGISTRY  # noqa: E402

KEYS = ("EVAL_LABEL", "EVAL_RETURN_MAP", "EVAL_UPGRADES")


def test_the_preset_is_registered_and_is_the_documented_configuration():
    assert REGISTRY["mb_gated_att"] == ("presets", "MbGatedAtt")
    assert presets.MB_GATED_ATT == {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1",
                                    "EVAL_UPGRADES": "bars,map_gated,attempt"}
    assert issubclass(presets.MbGatedAtt, presets.Action)


def test_apply_sets_the_flags_and_accepts_the_same_ones_in_any_order(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    presets.apply(presets.MB_GATED_ATT, "mb_gated_att")
    assert all(os.environ[k] == v for k, v in presets.MB_GATED_ATT.items())
    monkeypatch.setenv("EVAL_UPGRADES", "attempt, bars,map_gated")
    presets.apply(presets.MB_GATED_ATT, "mb_gated_att")        # same set: fine


def test_apply_refuses_a_conflicting_flag(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EVAL_LABEL", "change")
    with pytest.raises(SystemExit, match="EVAL_LABEL"):
        presets.apply(presets.MB_GATED_ATT, "mb_gated_att")
