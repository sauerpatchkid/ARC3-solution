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


def test_dev_sweep_2_needs_dev2_and_both_of_its_arms(monkeypatch):
    run_main(monkeypatch, "--arms", "A0-fresh,GB", "--games", "dev", "--seeds", "3,4,5,6",
             "--book", "pool_g0b_g1", "--dev2")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--arms", "GB", "--games", "dev", "--book", "pool_g0b_g1", "--dev2")


def planned(monkeypatch, *args, **kw):
    """The verdict files verdicts() would write, and with which rule."""
    calls = []
    monkeypatch.setattr(gb_sweep.subprocess, "run", lambda cmd, cwd=None: calls.append(cmd))
    gb_sweep.verdicts("/out", "/out/a0.manifest.tsv", *args, **kw)
    return {os.path.basename(c[c.index("--out") + 1]): ("--rule" in c and c[c.index("--rule") + 1]) for c in calls}


def test_dev_sweep_2s_verdict_is_gb_against_the_same_day_a0(monkeypatch):
    # seeds 3-6: A0's 27 Sep runs have none of them, so nothing is compared with them
    assert planned(monkeypatch, 32, 0, ["A0-fresh", "GB"], dev2=True) == {"verdict_gb_vs_a0_fresh.md": "dev"}
    # G2 as it ran: GB and GB-free against A0's runs under the dev rule
    assert planned(monkeypatch, 16, 16, ["GB", "GB-free"]) == {
        "verdict_gb_vs_a0.md": "dev", "gb_free_vs_a0.md": "dev", "gb_vs_gb_free.md": False}
