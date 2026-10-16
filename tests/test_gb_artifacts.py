"""The artifact contract (custom_agents/gb/artifacts.py): a stage is skipped only
when its version, its settings and every input file are unchanged."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import artifacts as A  # noqa: E402


@pytest.fixture
def run(tmp_path):
    return A.Run("toy", root=str(tmp_path))


def _file(path, text):
    with open(path, "w") as f:
        f.write(text)
    return str(path)


def test_the_settings_file_has_one_section_per_stage_and_helper():
    config = A.load_config()
    assert {"run", "see", "write", "check", "carry", "act", "schedule", "vocab", "templates", "show"} <= set(config)
    assert config["check"]["trusted_min_cases"] == 20 and config["write"]["answer_tokens_online"] == 12288


def test_a_run_records_its_settings_and_each_session(tmp_path):
    A.Run("toy", root=str(tmp_path))
    r = A.Run("toy", root=str(tmp_path))                       # a second session on the same run
    manifest = json.load(open(os.path.join(r.dir, "manifest.json")))
    assert len(manifest["sessions"]) == 2
    assert manifest["sessions"][0]["config_hash"] == A.digest(A.load_config())
    assert os.path.exists(os.path.join(r.dir, "config.toml"))


def test_a_stage_is_current_only_for_the_same_version_settings_and_inputs(run, tmp_path):
    src = _file(tmp_path / "evidence.npz", "level 1")
    out = _file(tmp_path / "facts.jsonl", "facts")
    settings = {"connectivity": 4}
    assert not run.is_current("s1_see", "ft09", 1, settings, {"evidence": src})
    run.write_header("s1_see", "ft09", 1, settings, {"evidence": src}, {"facts": out})
    assert run.is_current("s1_see", "ft09", 1, settings, {"evidence": src})
    assert not run.is_current("s1_see", "ft09", 2, settings, {"evidence": src})              # version bumped
    assert not run.is_current("s1_see", "ft09", 1, {"connectivity": 8}, {"evidence": src})   # a knob changed
    assert not run.is_current("s1_see", "tu93", 1, settings, {"evidence": src})              # another game
    _file(tmp_path / "evidence.npz", "level 1, re-recorded")
    assert not run.is_current("s1_see", "ft09", 1, settings, {"evidence": src})              # an input changed


def test_a_consumer_refuses_a_different_version(run, tmp_path):
    src = _file(tmp_path / "in.txt", "x")
    run.write_header("s3_check", "m0r0", 1, {}, {"in": src}, {"in": src})
    assert run.require("s3_check", "m0r0", 1)["stage"] == "s3_check"
    with pytest.raises(ValueError):
        run.require("s3_check", "m0r0", 2)
    with pytest.raises(FileNotFoundError):
        run.require("s4_carry", "m0r0", 1)


def test_only_the_six_stages_have_folders(run):
    assert os.path.isdir(run.stage_dir("s6_show"))
    with pytest.raises(ValueError):
        run.stage_dir("s7_extra")
