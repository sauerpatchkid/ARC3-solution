"""tools/gb_sweep.py: each arm's environment, which arms may run on dev games, G4's run set and
verdicts, and pausing a sweep (--status, --pause, --stop)."""
import os
import signal
import subprocess
import sys

import pytest

import gb_sweep


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


# ---- pausing ----------------------------------------------------------------------------------------
def test_in_progress_forgets_the_runs_a_stop_cut_short():
    log = "\n".join([
        "=" * 72, " GLASSBOX ONLINE  2 arms (A0-fresh, GB) x 25 games x 3 seeds x 100,000 actions; book pool_g0b_g1",
        " runs to do: 150 (already done: 0)   parallel jobs: 4   ETA ~26.3 h (~162 act/s total)",
        "start A0-fresh tu93 s0  10:00:01", "start GB       tu93 s0  10:00:02", "start A0-fresh tr87 s0  10:00:04",
        "done A0-fresh tu93 s0  levels=4   2570s", "!! GB       tu93 s0: no recording (see x.log)",
        "start GB       tr87 s0  10:43:00"])
    assert gb_sweep.in_progress(log) == {("A0-fresh", "tr87", "s0"): "10:00:04", ("GB", "tr87", "s0"): "10:43:00"}
    log += "\n GLASSBOX ONLINE  2 arms\nstart A0-fresh tr87 s0  12:00:00"      # stopped, then resumed
    assert gb_sweep.in_progress(log) == {("A0-fresh", "tr87", "s0"): "12:00:00"}


def test_status_counts_runs_and_shows_no_levels(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gb_sweep.A, "results_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gb_sweep, "sweep_pids", lambda run: [1])
    out = tmp_path / "g4" / "online"
    out.mkdir(parents=True)
    (out / "manifest.tsv").write_text("/r/a\ttu93\t0\tA0-fresh\n/r/b\tbp35\t0\tGB\n")
    (out / "STOP").write_text("")
    (tmp_path / "g4" / "g4_log.txt").write_text("\n".join([
        " GLASSBOX ONLINE  2 arms", " runs to do: 150 (already done: 0)   parallel jobs: 4   ETA ~26.3 h",
        "start A0-fresh tu93 s0  10:00:01", "start GB       bp35 s0  10:00:02", "start GB       tu93 s0  10:00:03",
        "done A0-fresh tu93 s0  levels=4   2400s", "done GB       bp35 s0  levels=sealed   2700s"]))
    gb_sweep.status("g4", str(out))
    text = capsys.readouterr().out
    assert "g4: 2 runs done of 150; the sweep is RUNNING" in text and "pausing" in text
    assert "in progress: GB       tu93 s0  since 10:00:03" in text
    assert "about 26.2 h of running left" in text                      # 148 runs x 2,550 s / 4 at a time
    assert "levels" not in text


def test_pause_writes_the_stop_file_only_for_a_running_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(gb_sweep, "sweep_pids", lambda run: [])
    assert not gb_sweep.pause("g4", str(tmp_path)) and not (tmp_path / "STOP").exists()
    monkeypatch.setattr(gb_sweep, "sweep_pids", lambda run: [12345])
    assert gb_sweep.pause("g4", str(tmp_path)) and (tmp_path / "STOP").exists()


FAKE = ("import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(120)\n")


def fake_sweep(run, own_group=True):
    """A process whose arguments are a sweep's for `run`, with a child (like one of its runs)."""
    p = subprocess.Popen([sys.executable, "-c", FAKE, "tools/gb_sweep.py", "--run", run],
                         stdout=subprocess.PIPE, text=True, start_new_session=own_group)
    return p, int(p.stdout.readline())


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return False
    return stat[stat.rindex(")") + 2] != "Z"


def test_stop_kills_the_sweeps_whole_group_and_nothing_else(monkeypatch):
    real_killpg = os.killpg
    sweep, its_run = fake_sweep("test_stop_g4")
    other, other_run = fake_sweep("test_stop_g4x")                     # another sweep: must survive
    signalled = []

    def killpg(group, sig):
        assert group == sweep.pid, "only the stopped sweep's own group is ever signalled"
        signalled.append(sig)
        real_killpg(group, sig)
    monkeypatch.setattr(gb_sweep.os, "killpg", killpg)
    try:
        assert gb_sweep.sweep_pids("test_stop_g4") == [sweep.pid]
        assert gb_sweep.stop("test_stop_g4", wait=10)
        sweep.wait(timeout=10)
        assert signalled[0] == signal.SIGTERM and not alive(its_run)
        assert alive(other.pid) and alive(other_run)
        assert gb_sweep.sweep_pids("test_stop_g4") == [] and not gb_sweep.stop("test_stop_g4")
    finally:
        for p in (sweep, other):
            try:
                real_killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()


def test_stop_refuses_a_sweep_that_does_not_lead_its_own_group(monkeypatch):
    sweep, its_run = fake_sweep("test_stop_shared", own_group=False)  # in pytest's group: hands off
    monkeypatch.setattr(gb_sweep.os, "killpg", lambda group, sig: pytest.fail("signalled a shared group"))
    try:
        with pytest.raises(SystemExit):
            gb_sweep.stop("test_stop_shared")
        assert alive(sweep.pid) and alive(its_run)
    finally:
        os.kill(its_run, signal.SIGKILL)
        sweep.kill()
        sweep.wait()
