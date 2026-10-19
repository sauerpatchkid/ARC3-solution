"""tools/gb_sweep.py: each arm's environment, and which arms may run on dev games."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gb_sweep  # noqa: E402


def test_arm_envs():
    gb = gb_sweep.arm_env("GB", "pool_g0b_g1")
    assert gb["EVAL_GB"] == "1" and os.path.isabs(gb["EVAL_GB_BOOK"])
    assert gb_sweep.arm_env("GB-free", "") == {"EVAL_GB": "1"}
    assert gb_sweep.arm_env("A0-fresh", "") == {}
    with pytest.raises(SystemExit):
        gb_sweep.arm_env("A1", "")


def test_no_arm_inherits_the_shells_eval_gb():
    shell = {"PATH": "/bin", "EVAL_GB": "1", "EVAL_GB_BOOK": "/somewhere/else", "EVAL_SEED": "3"}
    a0 = gb_sweep.run_env("A0-fresh", "", shell)
    assert "EVAL_GB" not in a0 and "EVAL_GB_BOOK" not in a0
    assert a0["PATH"] == "/bin" and a0["EVAL_SEED"] == "3"
    assert "EVAL_GB_BOOK" not in gb_sweep.run_env("GB-free", "", shell)


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["gb_sweep.py", "--run", "test_gb_sweep", "--dry-run", *args])
    gb_sweep.main()


def test_dev_games_need_g2_only_with_s5(monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--arms", "GB-free", "--games", "dev")
    run_main(monkeypatch, "--arms", "A0-fresh", "--games", "dev")       # no S5: allowed
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--arms", "A0-fresh", "--games", "bp35")  # held-out: never
