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


G4_ARGS = ("--arms", "A0-fresh,GB", "--games", "all", "--seeds", "0,1,2", "--book", "pool_g0b_g1", "--g4")


def test_held_out_games_run_only_inside_g4(monkeypatch):
    for games in ("bp35", "ar25", "all"):
        with pytest.raises(SystemExit):
            run_main(monkeypatch, "--arms", "A0-fresh", "--games", games)
    with pytest.raises(SystemExit):                                    # nor under another gate's flag
        run_main(monkeypatch, "--arms", "A0-fresh,GB", "--games", "all", "--seeds", "0,1,2",
                 "--book", "pool_g0b_g1", "--dev2")


def test_g4_is_the_whole_run_set_or_nothing(monkeypatch, capsys):
    assert gb_sweep.ALL[:8] == gb_sweep.tiers.DEV and len(set(gb_sweep.ALL)) == 25   # tier by tier
    run_main(monkeypatch, *G4_ARGS)
    assert "runs to do: 150" in capsys.readouterr().out                # 25 games x 3 seeds x 2 arms
    partial = [("--games", "dev"), ("--games", "ls20,bp35"), ("--seeds", "0,1"), ("--seeds", "0,1,2,3"),
               ("--arms", "GB"), ("--arms", "GB,GB-free"), ("--arms", "A0-fresh,GB,GB-free"), ("--cap", "20000")]
    for flag, value in partial:
        args = list(G4_ARGS)
        if flag in args:
            args[args.index(flag) + 1] = value
        else:
            args += [flag, value]
        with pytest.raises(SystemExit):
            run_main(monkeypatch, *args)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, *G4_ARGS, "--dev2")


def test_g4s_verdict_is_the_confirm_rule_against_the_same_day_a0_with_held_out_rows_sealed(monkeypatch):
    calls = []
    monkeypatch.setattr(gb_sweep.subprocess, "run", lambda cmd, cwd=None: calls.append(cmd))
    gb_sweep.verdicts("/out", "/out/a0.manifest.tsv", 75, 75, ["A0-fresh", "GB"], g4=True)
    got = {os.path.basename(c[c.index("--out") + 1]): c for c in calls}
    assert list(got) == ["verdict_gb_vs_a0_fresh.md", "gb_vs_a0.md", "a0_fresh_vs_a0.md"]
    v = got["verdict_gb_vs_a0_fresh.md"]
    assert v[v.index("--base") + 1] == "/out/manifest.tsv:A0-fresh" and v[v.index("--new") + 1] == "/out/manifest.tsv:GB"
    assert v[v.index("--rule") + 1] == "confirm" and v[v.index("--expect") + 1] == "75"
    assert got["gb_vs_a0.md"][got["gb_vs_a0.md"].index("--rule") + 1] == "confirm"     # 112 beside it, same rule
    assert "--rule" not in got["a0_fresh_vs_a0.md"]                                     # drift: no rule
    seals = [c[c.index("--seal") + 1] for c in calls]                                   # every one sealed, apart
    assert seals == ["/out/verdict_gb_vs_a0_fresh.sealed", "/out/gb_vs_a0.sealed", "/out/a0_fresh_vs_a0.sealed"]
    # without --g4 nothing is sealed (dev sweep 2 as it ran)
    calls.clear()
    gb_sweep.verdicts("/out", "/out/a0.manifest.tsv", 32, 0, ["A0-fresh", "GB"], dev2=True)
    assert all("--seal" not in c for c in calls)


def test_g4s_log_never_shows_a_held_out_games_levels():
    for game in gb_sweep.tiers.HELD_OUT:
        line = gb_sweep.done_line("GB", game, "1", 3, 2570.4, g4=True)
        assert "levels=sealed" in line and "levels=3" not in line
    for game in gb_sweep.tiers.DEV + gb_sweep.tiers.SEEN:                               # dev and seen as before
        assert "levels=3" in gb_sweep.done_line("GB", game, "1", 3, 2570.4, g4=True)
    assert gb_sweep.done_line("A0-fresh", "vc33", "3", 4, 2570.4) == "done A0-fresh vc33 s3  levels=4   2570s"
