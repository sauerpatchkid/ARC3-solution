"""experiments/goose/screen3.py and screen3_report.py: the run set, and the rule on made-up runs."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "goose"))
import screen3 as S  # noqa: E402
import screen3_report as R  # noqa: E402
import tiers  # noqa: E402
from upgrades import Upgrades  # noqa: E402


# ---- the run set -----------------------------------------------------------------------------------
def test_the_arms_are_the_nine_in_the_plan():
    flags = {arm: (S.ARMS[arm][0], S.ARMS[arm][1]) for arm in S.TABLE_ORDER}
    train = {"EVAL_TRAIN_EVERY": str(S.TRAIN_EVERY), "EVAL_TRAIN_RECENT": "16"}
    assert flags == {
        "base": ("bars,map_gated,attempt", {}),
        "S": ("bars,map_sched,attempt", {}),
        "O": ("bars,map_gated,attempt,objclick", {}),
        "T": ("bars,map_gated,attempt", train),
        "SO": ("bars,map_sched,attempt,objclick", {}),
        "ST": ("bars,map_sched,attempt", train),
        "OT": ("bars,map_gated,attempt,objclick", train),
        "SOT": ("bars,map_sched,attempt,objclick", train),
        "base2": ("bars,map_gated,attempt", {}),
    }
    assert S.TRAIN_EVERY in (1, 2)
    assert list(S.ARMS) == ["base", "S", "O", "T", "SOT", "SO", "ST", "OT", "base2"]      # the order within a game
    assert S.UPGRADED == ["S", "O", "T", "SO", "ST", "OT", "SOT"]
    for arm, (upgrades, _, has) in S.ARMS.items():
        Upgrades(upgrades.split(","))                                   # every list is one the agent accepts
        assert ("map_sched" in upgrades) == ("S" in has) == ("map_gated" not in upgrades)
        assert ("objclick" in upgrades) == ("O" in has)
        assert ("EVAL_TRAIN_EVERY" in S.arm_env(arm, 5000)) == ("T" in has)
    env = S.arm_env("SOT", 5000)
    assert (env["EVAL_LABEL"], env["EVAL_RETURN_MAP"], env["EVAL_SEED"], env["EVAL_MAX_ACTIONS"], env["PYTHONHASHSEED"]) \
        == ("novel", "1", "0", "5000", "0")


@pytest.mark.skipif(S.TRIAL, reason="TRIAL=1 shrinks the run set")
def test_the_queue_is_225_runs_games_outside_with_the_tiers_taken_in_turn():
    jobs = S.sweep_jobs()
    assert len(jobs) == 225 == len(set(jobs)) and {j.cap for j in jobs} == {5000}
    games = list(dict.fromkeys(j.game for j in jobs))
    assert len(games) == 25
    assert [tiers.tier_of(g) for g in games[:8]] == ["dev", "seen", "ab_offline", "untouched"] * 2
    assert [j.arm for j in jobs[:9]] == list(S.ARMS) and {j.game for j in jobs[:9]} == {games[0]}


def test_the_callers_eval_settings_never_reach_a_run(monkeypatch):
    monkeypatch.setenv("EVAL_GB", "1")
    monkeypatch.setenv("EVAL_MAP_STALL", "7")
    assert not [k for k in S.clean_env() if k.startswith("EVAL_")]


# ---- the rule, on made-up runs ---------------------------------------------------------------------
GAMES = ("ft09", "tu93", "ls20")


def make_sweep(tmp_path, levelups, skip=()):
    """A sweep folder with one run per (arm, game); `levelups(arm, game)` gives the level-up actions."""
    rows = []
    for arm in S.ARMS:
        for g in GAMES:
            if (arm, g) in skip:
                continue
            ups = levelups(arm, g)
            d = tmp_path / "runs" / arm / g
            d.mkdir(parents=True)
            (d / "metrics.json").write_text(json.dumps({
                "levels_completed": len(ups), "levelup_events": [[a, i + 1] for i, a in enumerate(ups)],
                "actions_taken": S.CAP, "n_actions": S.CAP - 20, "unique_states": 100 + len(ups), "wall_ms_med": 8.0,
                "termination": "cap"}))
            (d / "run_end.json").write_text(json.dumps({"actions_per_sec": 100.0, "termination": "cap"}))
            (d / "run_config.json").write_text(json.dumps(dict(
                upgrades=",".join(sorted(S.ARMS[arm][0].split(","))), label="novel", return_map=True, max_actions=S.CAP,
                seed_source="env", reset_on_level=True, train_frequency=int(S.ARMS[arm][1].get("EVAL_TRAIN_EVERY", 5)),
                **({"train_every": S.TRAIN_EVERY, "train_recent": 16} if S.ARMS[arm][1] else {}))))
            rows.append(f"{d}\t{g}\t0\t{arm}")
    (tmp_path / "manifest.tsv").write_text("\n".join(rows) + "\n")
    (tmp_path / "sweep.json").write_text(json.dumps({"commit": "abc1234", "jobs": 4, "offline": True, "started": "now"}))
    return str(tmp_path)


def s_helps(arm, g):
    """One level everywhere; arms with S finish a second one on every game."""
    return [100, 900] if "S" in S.ARMS[arm][2] else [100]


def test_an_arm_that_wins_on_every_game_clears_the_noise_and_the_main_effects_say_which_upgrade(tmp_path):
    out = make_sweep(tmp_path, s_helps)
    runs = R.load(out)
    assert R.total(runs, "base", GAMES) == 3 and R.total(runs, "S", GAMES) == 6 and R.total(runs, "S", GAMES, 500) == 3
    v = R.against_base(runs, "S", GAMES)
    assert (v["delta"], v["lo"], v["hi"], v["better"], v["same"], v["worse"]) == (3, 3, 3, 3, 0, 0)
    assert R.clears_noise(6, 3, 3, v["lo"]) and not R.clears_noise(6, 3, 6, v["lo"]) and not R.clears_noise(6, 3, 3, 0)
    assert R.rank(runs, GAMES) == ["S", "SO", "ST", "SOT", "O", "T", "OT"]     # ties: the table's order
    eff = {u: R.main_effect(runs, u, GAMES) for u in "SOT"}
    assert eff["S"]["effect"] == 3 and eff["O"]["effect"] == 0 and eff["T"]["effect"] == 0
    assert sorted(eff["S"]["yes"]) == ["S", "SO", "SOT", "ST"] and "base2" not in eff["S"]["no"]
    assert R.effects_agree("S", eff) and not R.effects_agree("SO", eff)
    assert R.added_to(runs, "S", GAMES) == [("base", "S", 3), ("O", "SO", 3), ("T", "ST", 3), ("OT", "SOT", 3)]
    assert R.added_to(runs, "O", GAMES) == [("base", "O", 0), ("S", "SO", 0), ("T", "OT", 0), ("ST", "SOT", 0)]
    assert R.against_both(runs, "S", GAMES) == (3.0, 3.0, 3.0)
    short = R.write(out)
    assert "Ranked first of the upgraded arms: `S`" in short and "clears the noise" in short
    report = open(os.path.join(out, "report.md")).read()
    assert "Runs in the manifest: **27 of" in report and "as the arm table says, on every run" in report
    assert len(open(os.path.join(out, "runs.csv")).read().splitlines()) == 28       # a header and one row per run


def test_ties_go_to_levels_by_2000_then_to_games_better_and_a_split_result_is_inside_the_noise(tmp_path):
    def levelups(arm, g):
        if arm == "O":                              # +1 on ft09, late; -1 on tu93: no better in total
            return {"ft09": [100, 4000], "tu93": [], "ls20": [100]}[g]
        if arm == "T":                              # +1 on ft09 by action 1,500
            return [100, 1500] if g == "ft09" else [100]
        if arm == "SO":                             # +1 on ft09, late
            return [100, 4500] if g == "ft09" else [100]
        return [100]
    out = make_sweep(tmp_path, levelups)
    runs = R.load(out)
    order = R.rank(runs, GAMES)
    assert order[:2] == ["T", "SO"]                 # both score 4; T has more levels by 2,000
    assert R.total(runs, "O", GAMES) == 3 and order.index("O") > 1
    v = R.against_base(runs, "T", GAMES)
    assert v["delta"] == 1 and v["lo"] == 0         # one game of three: the interval reaches 0
    assert not R.clears_noise(R.total(runs, "T", GAMES), 3, 3, v["lo"])
    assert "inside the noise" in R.write(out) and "**none**" in R.write(out)


def test_a_partial_sweep_is_compared_on_the_games_every_arm_has_and_says_what_is_missing(tmp_path):
    out = make_sweep(tmp_path, s_helps, skip={("OT", "ls20"), ("base2", "ls20")})
    short = R.write(out)
    report = open(os.path.join(out, "report.md")).read()
    assert "PARTIAL SWEEP: 25 of" in short and "(tu93, ft09)" in report
    assert "OT ls20" in report and "base2 ls20" in report
    assert R.total(R.load(out), "S", ("ft09", "tu93")) == 4


def test_a_run_with_the_wrong_flags_is_named(tmp_path):
    out = make_sweep(tmp_path, s_helps)
    cfg = os.path.join(out, "runs", "T", "ft09", "run_config.json")
    c = json.load(open(cfg))
    c["train_every"] = 5
    json.dump(c, open(cfg, "w"))
    R.write(out)
    assert "NOT as the arm table says on: T ft09" in open(os.path.join(out, "report.md")).read()


# ---- the runner: a failed run is skipped, tried once more, and never fatal ------------------------------
def fake_sweep(tmp_path, monkeypatch, fails):
    """work() on six made-up jobs; `fails(job, attempt)` says whether a try of a job fails."""
    jobs = [S.Job(g, a, 300, a) for g in ("ft09", "tu93") for a in ("base", "S", "base2")]
    tries, reports = {}, []

    def run_job(out, job, lock):
        tries[job] = tries.get(job, 0) + 1
        if fails(job, tries[job]):
            if job.arm == "S" and job.game == "tu93":
                raise RuntimeError("the run blew up")           # an exception counts as a failed run
            return "no recording (the run crashed: see its log)"
        with lock, open(os.path.join(out, "manifest.tsv"), "a") as f:
            f.write(f"somewhere/{job.label}/{job.game}\t{job.game}\t0\t{job.label}\n")
        return None

    monkeypatch.setattr(S, "sweep_jobs", lambda: jobs)
    monkeypatch.setattr(S, "run_job", run_job)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    monkeypatch.setitem(sys.modules, "screen3_report",
                        type("FakeReport", (), {"write": staticmethod(lambda out, docs=False: reports.append(out))}))
    (tmp_path / "sweep.json").write_text(json.dumps({"jobs": 2}))
    S.work(str(tmp_path))
    return tries, reports


def test_a_failed_run_is_logged_tried_once_more_and_the_report_is_written(tmp_path, monkeypatch):
    tries, reports = fake_sweep(tmp_path, monkeypatch, lambda job, attempt: job.arm == "S" and attempt == 1)
    assert sorted(tries.values()) == [1, 1, 1, 1, 2, 2]                 # the two S runs: once more
    assert len(S.done_runs(str(tmp_path))) == 6 and reports == [str(tmp_path)]
    failed = S.read_tsv(str(tmp_path), "failed.tsv")
    assert sorted((r[0], r[1]) for r in failed) == [("ft09", "S"), ("tu93", "S")]
    assert any("RuntimeError" in r[3] for r in failed)


def test_a_run_that_keeps_failing_is_left_out_and_the_report_is_still_written(tmp_path, monkeypatch):
    tries, reports = fake_sweep(tmp_path, monkeypatch, lambda job, attempt: job == S.Job("ft09", "S", 300, "S"))
    assert tries[S.Job("ft09", "S", 300, "S")] == 2                     # once, and once more: not a third time
    assert len(S.done_runs(str(tmp_path))) == 5 and reports == [str(tmp_path)]


def test_a_stop_file_pauses_the_queue_and_a_paused_sweep_writes_no_report(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"

    def fails(job, attempt):
        stop.write_text("")                                             # a pause is asked for during the first run
        return False
    tries, reports = fake_sweep(tmp_path, monkeypatch, fails)
    assert 1 <= len(tries) < 6 and reports == [] and not stop.exists()  # the STOP file is used up
    assert len(S.done_runs(str(tmp_path))) == len(tries)                # what had started was finished and kept
