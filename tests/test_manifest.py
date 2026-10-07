"""The one manifest reader (manifest.py)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifest import arm_name, read_manifest  # noqa: E402


def test_rows_keep_file_order_and_text_seeds(tmp_path):
    p = tmp_path / "m.manifest"
    p.write_text("runs/a\tft09\t0\tA1\n\nruns/b\ttu93\t2\ton\n")
    assert read_manifest(p) == [
        {"run_dir": "runs/a", "game": "ft09", "seed": "0", "arm": "A1"},
        {"run_dir": "runs/b", "game": "tu93", "seed": "2", "arm": "on"}]


def test_a_cut_off_line_is_skipped_unless_strict(tmp_path):
    p = tmp_path / "m.manifest"
    p.write_text("runs/a\tft09\t0\tA1\nruns/b\ttu9")          # killed mid-write
    assert [r["game"] for r in read_manifest(p)] == ["ft09"]
    with pytest.raises(ValueError, match="m.manifest:2"):
        read_manifest(p, strict=True)


def test_arm_names():
    assert arm_name("on") == "reset_on" and arm_name("off") == "reset_off"
    assert arm_name("A1") == "A1" and arm_name("mb_gated_att") == "mb_gated_att"
