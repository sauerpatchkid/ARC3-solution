"""experiments/kp/kp_verdict.py: the test's rule, on made-up runs."""
import json
import os
import sys

import pytest

pytest.importorskip("arc_agi")      # kp_test's settings hash loads the bridge, which loads the reference

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "kp"))
import kp_test as T  # noqa: E402
import kp_verdict as V  # noqa: E402

GAMES = ("ft09", "lp85", "tu93", "vc33")        # tu93 has no click action
SOT_CFG = {"label": "novel", "return_map": True, "train_every": 1, "train_recent": 16,
           "upgrades": "attempt,bars,map_sched,objclick"}


def sweep(tmp_path, monkeypatch, levels, skip=()):
    """A made-up sweep of four games x two seeds x three arms. levels[arm][game] = (seed 11, seed 12)."""
    jobs = [T.Job(g, s, a, 5000) for g in GAMES for s in (11, 12) for a in ("SOT", "SOT-KP", "SOT2")]
    monkeypatch.setattr(T, "sweep_jobs", lambda: jobs)
    monkeypatch.setattr(V, "ROOT", str(tmp_path))
    rows = []
    for j in jobs:
        if (j.game, j.seed, j.arm) in skip:
            continue
        n = levels[j.arm][j.game][j.seed - 11]
        d = tmp_path / "runs" / j.arm / f"{j.game}_s{j.seed}"
        (d / "kp").mkdir(parents=True)
        ups = [[400 * (k + 1), k + 1] for k in range(n)]
        planned = j.arm == "SOT-KP" and j.game != "tu93"
        files = {"metrics.json": {"levels_completed": n, "levelup_events": ups},
                 "run_end.json": {"n_actions": 5000, "elapsed_s": 500.0, "termination": "cap"},
                 "run_config.json": dict(SOT_CFG, **({"kp_config_hash": T.kp_config_hash()} if j.arm == "SOT-KP" else {})),
                 "train_stats.json": {"train_steps": 4800}}
        if j.arm == "SOT-KP":
            kp_levels = [dict(level=k + 1, entered_at=400 * k, left_at=400 * (k + 1), screens=400, screens_that_fit=400,
                              clicks={}, gave_up="", ended="cleared by planning" if k else "cleared by Goose")
                         for k in range(n)]
            files["kp/kp_stats.json"] = dict(
                dormant=not planned, kp_errors=0, levels=kp_levels if planned else [],
                levels_by_planning=max(0, n - 1) if planned else 0, levels_by_goose=min(n, 1) if planned else 0,
                clicks={"probe": 9, "verify": 0, "plan": 5})
            (d / "kp" / "events.jsonl").write_text("".join(
                json.dumps({"step": 1, "level": k + 1, "event": "reference", "entry": [kind, k]}) + "\n"
                for k in range(1, n) for kind in ("probe-set", "goal", "plan")))
        for name, content in files.items():
            (d / name).write_text(json.dumps(content))
        rows.append(f"runs/{j.arm}/{j.game}_s{j.seed}\t{j.game}\t{j.seed}\t{j.arm}\n")
    (tmp_path / "manifest.tsv").write_text("".join(rows))
    (tmp_path / "environment_files" / "ft09" / "0d8bbf25").mkdir(parents=True)     # the game's own human baseline
    (tmp_path / "environment_files" / "ft09" / "0d8bbf25" / "metadata.json").write_text(
        json.dumps({"baseline_actions": [43, 12, 23, 28, 65, 37]}))
    return str(tmp_path)


BASE = {"ft09": (1, 1), "lp85": (1, 2), "tu93": (2, 3), "vc33": (2, 2)}      # 14 levels
AGAIN = {"ft09": (1, 2), "lp85": (2, 1), "tu93": (3, 2), "vc33": (2, 2)}     # 15: differs by one level on five pairs


def test_the_chance_gap_is_the_same_on_every_run_and_grows_with_the_differences():
    small, large = V.chance_gap([0, 0, 1, -1, 0, 1]), V.chance_gap([0, 0, 3, -3, 0, 3])
    assert small == V.chance_gap([0, 0, 1, -1, 0, 1]) and 0 < small < large
    assert V.chance_gap([0] * 50) == 0                                  # two identical baselines: any gain is above it


def test_a_planner_far_ahead_of_both_baselines_clears_the_noise(tmp_path, monkeypatch):
    out = sweep(tmp_path, monkeypatch, {"SOT": BASE, "SOT2": AGAIN,
                                        "SOT-KP": {"ft09": (6, 6), "lp85": (4, 4), "tu93": (2, 3), "vc33": (2, 2)}})
    runs = V.load(out)
    pairs = sorted(runs["SOT"])
    r = V.rule(runs, pairs)
    assert r["score"] == {"SOT": 14, "SOT-KP": 29, "SOT2": 15} and r["gap"] == 15 and r["above_both"] and r["clears"]
    assert 0 < r["bar"] < 15
    text = V.write(out)
    assert "**The planner: CLEARS THE NOISE.**" in text and "| SOT-KP | 29 |" in text
    assert "| ft09 | 2 | 10 | 2 |" in text                              # ft09, two runs: ten levels by planning, two by Goose
    assert "| tu93 | 2 | dormant: no click action" in text
    assert "L2: 400 (2/2 runs, human 12)" in text                       # beside the game's own human baseline
    assert os.path.exists(os.path.join(out, "runs.csv"))


def test_a_planner_ahead_by_less_than_two_baselines_differ_is_inside_the_noise(tmp_path, monkeypatch):
    swing = {"ft09": (3, 0), "lp85": (0, 3), "tu93": (4, 1), "vc33": (3, 2)}             # 16: the baselines swing a lot
    out = sweep(tmp_path, monkeypatch, {"SOT": BASE, "SOT2": swing,
                                        "SOT-KP": {"ft09": (2, 2), "lp85": (2, 2), "tu93": (2, 3), "vc33": (2, 3)}})
    r = V.rule(V.load(out), sorted(V.load(out)["SOT"]))
    assert r["score"]["SOT-KP"] == 18 and r["above_both"] and r["gap"] == 4 <= r["bar"] and not r["clears"]
    assert "INSIDE THE NOISE" in V.write(out)


def test_a_planner_below_the_second_baseline_does_not_clear_whatever_its_gap(tmp_path, monkeypatch):
    high = {"ft09": (2, 2), "lp85": (2, 2), "tu93": (3, 3), "vc33": (3, 3)}              # 20
    out = sweep(tmp_path, monkeypatch, {"SOT": BASE, "SOT2": high,
                                        "SOT-KP": {"ft09": (2, 2), "lp85": (2, 2), "tu93": (2, 3), "vc33": (3, 3)}})
    r = V.rule(V.load(out), sorted(V.load(out)["SOT"]))
    assert r["score"]["SOT-KP"] == 19 > r["score"]["SOT"] and not r["above_both"] and not r["clears"]


def test_no_verdict_before_every_run_is_in(tmp_path, monkeypatch):
    out = sweep(tmp_path, monkeypatch, {"SOT": BASE, "SOT2": AGAIN, "SOT-KP": BASE}, skip={("vc33", 12, "SOT2")})
    assert "No verdict and no table: 1 runs are not in the manifest yet" in V.write(out)
    text = V.write(out, partial=True)
    assert "NOT GIVEN: the sweep is not complete" in text and "7 of 8 game-and-seed pairs" in text


def test_pairs_and_games_worse_on_every_seed_are_counted(tmp_path, monkeypatch):
    out = sweep(tmp_path, monkeypatch, {"SOT": BASE, "SOT2": AGAIN,
                                        "SOT-KP": {"ft09": (3, 3), "lp85": (1, 2), "tu93": (2, 3), "vc33": (1, 1)}})
    runs = V.load(out)
    v = V.against(runs, "SOT-KP", "SOT", sorted(runs["SOT"]))
    assert (v["delta"], v["better"], v["same"], v["worse"]) == (2, 2, 4, 2) and v["worse_on_every_seed"] == ["vc33"]
    assert v["lo"] <= 2 <= v["hi"]
