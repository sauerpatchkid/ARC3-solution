"""experiments/kp/kp_test.py: the run sets of the smoke, the dev sweep and the test, and the queue."""
import importlib
import json
import os
import sys

import pytest

pytest.importorskip("arc_agi")      # kp_config_hash() loads the bridge, which loads the reference

for _folder in ("goose", "kp"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", _folder))
import kp_bridge  # noqa: E402
import kp_test as T  # noqa: E402
import screen3  # noqa: E402
import tiers  # noqa: E402


# ---- the run sets ------------------------------------------------------------------------------------
def test_the_base_flags_are_upgrade_screen_3s_sot_arm():
    theirs = {k: v for k, v in screen3.arm_env("SOT", 5000).items() if k != "EVAL_SEED"}
    for seed in T.SEEDS:
        ours = T.arm_env("SOT", seed)
        assert ours.pop("EVAL_SEED") == str(seed) and ours == theirs


def test_the_three_arms_differ_only_by_the_planner():
    sot, kp, sot2 = (T.arm_env(arm, 11) for arm in ("SOT", "SOT-KP", "SOT2"))
    assert sot == sot2                                                  # the baseline again is the baseline
    assert {k: v for k, v in kp.items() if k not in sot} == {"EVAL_KP": "1", "EVAL_KP_CONFIG": "experiments/kp/kp.toml"}
    assert all(kp[k] == v for k, v in sot.items())
    assert T.kp_config_hash() == kp_bridge.digest(kp_bridge.DEFAULTS)    # the committed settings are the reference's


def test_the_test_is_150_runs_games_outside_with_the_tiers_taken_in_turn():
    jobs = T.sweep_jobs()
    assert len(jobs) == 150 == len(set(jobs)) and {j.cap for j in jobs} == {5000} and {j.seed for j in jobs} == {11, 12}
    games = list(dict.fromkeys(j.game for j in jobs))
    assert len(games) == 25 and [tiers.tier_of(g) for g in games[:8]] == ["dev", "seen", "ab_offline", "untouched"] * 2
    assert [(j.seed, j.arm) for j in jobs[:6]] == [(s, a) for s in (11, 12) for a in ("SOT", "SOT-KP", "SOT2")]
    assert {j.game for j in jobs[:6]} == {games[0]}                     # the arms run back to back inside a game and seed


def test_the_dev_sweep_is_28_runs_on_the_carry_tests_seven_games(monkeypatch):
    monkeypatch.setenv("DEV", "1")
    dev = importlib.reload(T)
    try:
        jobs = dev.sweep_jobs()
        assert len(jobs) == 28 and {j.seed for j in jobs} == {14, 15} and {j.arm for j in jobs} == {"SOT", "SOT-KP"}
        assert {j.game for j in jobs} == {"vc33", "ft09", "cd82", "dc22", "m0r0", "lp85", "tu93"} <= set(tiers.BUILD_FROM)
        assert dev.OUT_ROOT == "results/kp/dev"
    finally:
        monkeypatch.delenv("DEV")
        importlib.reload(T)
    assert T.OUT_ROOT == "results/kp/test"


def test_no_seed_of_these_runs_was_used_before():
    used = {11, 12, 14, 15}                 # the test, the dev sweep and the smoke
    assert set(T.SEEDS) | {T.SMOKE_SEED} <= used and not used & set(range(11))      # 0 to 10: G4, dev sweep 2, the carry test, screen 3


def test_the_callers_eval_settings_never_reach_a_run(monkeypatch):
    monkeypatch.setenv("EVAL_GB", "1")
    monkeypatch.setenv("EVAL_KP_PLAN", "0")
    assert not [k for k in T.clean_env() if k.startswith("EVAL_")]


def test_the_code_hash_changes_when_a_file_the_runs_depend_on_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "ROOT", str(tmp_path))
    monkeypatch.setattr(T, "CODE", ["agents", "settings.toml"])
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "a.py").write_text("x = 1\n")
    (tmp_path / "agents" / "notes.md").write_text("not code\n")
    (tmp_path / "settings.toml").write_text("[bridge]\nplan = true\n")
    first = T.code_hash()
    (tmp_path / "agents" / "notes.md").write_text("still not code\n")
    assert T.code_hash() == first
    (tmp_path / "settings.toml").write_text("[bridge]\nplan = false\n")
    assert T.code_hash() != first


# ---- the queue ---------------------------------------------------------------------------------------
def fake_sweep(tmp_path, monkeypatch, fails):
    """work() on eight made-up jobs; `fails(job, attempt)` says whether a try of a job fails."""
    jobs = [T.Job(g, s, a, 300) for g in ("ft09", "tu93") for s in (11, 12) for a in ("SOT", "SOT-KP")]
    tries = {}

    def run_job(out, job, lock):
        tries[job] = tries.get(job, 0) + 1
        if fails(job, tries[job]):
            return "no recording (the run crashed: see its log)"
        with lock, open(os.path.join(out, "manifest.tsv"), "a") as f:
            f.write(f"somewhere/{job.arm}/{job.game}_s{job.seed}\t{job.game}\t{job.seed}\t{job.arm}\n")
        return None

    monkeypatch.setattr(T, "sweep_jobs", lambda: jobs)
    monkeypatch.setattr(T, "run_job", run_job)
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    (tmp_path / "sweep.json").write_text(json.dumps({"jobs": 2}))
    T.work(str(tmp_path))
    return tries


def test_a_failed_run_is_logged_and_tried_once_more(tmp_path, monkeypatch):
    tries = fake_sweep(tmp_path, monkeypatch, lambda job, attempt: job.arm == "SOT-KP" and job.seed == 12)
    assert sorted(tries.values()) == [1] * 6 + [2, 2]                   # once, and once more: not a third time
    done = T.done_runs(str(tmp_path))
    assert len(done) == 6 and ("ft09", 11, "SOT-KP") in done and ("ft09", 12, "SOT-KP") not in done
    assert sorted({(r[0], r[1], r[2]) for r in T.read_tsv(str(tmp_path), "failed.tsv")}) == [
        ("ft09", "12", "SOT-KP"), ("tu93", "12", "SOT-KP")]


def test_a_stop_file_pauses_the_queue_and_keeps_what_was_finished(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"

    def fails(job, attempt):
        stop.write_text("")                                             # a pause is asked for during the first run
        return False
    tries = fake_sweep(tmp_path, monkeypatch, fails)
    assert 1 <= len(tries) < 8 and not stop.exists()                    # the STOP file is used up
    assert len(T.done_runs(str(tmp_path))) == len(tries)


def test_a_resume_refuses_changed_code_and_changed_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "current", lambda must_exist=True: str(tmp_path))
    monkeypatch.setattr(T, "launch", lambda out: (_ for _ in ()).throw(AssertionError("it must not start")))
    arms = {arm: T.arm_env(arm, "<seed>") for arm in T.ARM_NAMES}
    for meta, why in (({"arms": arms, "code_hash": "another", "commit": "abcdef0"}, "code the runs depend on changed"),
                      ({"arms": dict(arms, SOT={}), "code_hash": T.code_hash(), "commit": "abcdef0"}, "arms")):
        (tmp_path / "sweep.json").write_text(json.dumps(meta))
        with pytest.raises(SystemExit, match=why):
            T.cmd_resume(None)
