"""experiments/kp/kp_explorer_verdict.py: the explorer's test's two rules, on made-up runs."""
import json
import os
import sys

import pytest

pytest.importorskip("arc_agi")      # kp_test's settings hash loads the bridge, which loads the reference

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "kp"))
import benchmark  # noqa: E402
import kp_bridge  # noqa: E402
import kp_explorer_verdict as X  # noqa: E402
import kp_test as T  # noqa: E402
import kp_verdict as V  # noqa: E402

ARMS = ("SOT-KP", "SOT-KP21", "SOT-KP2")
GAMES = ("cd82", "tu93", "sp80", "ft09")        # three with keys (one of them held-out), one with clicks only
SOT_CFG = {"label": "novel", "return_map": True, "train_every": 1, "train_recent": 16,
           "upgrades": "attempt,bars,map_sched,objclick"}
E21_HASH = kp_bridge.digest(kp_bridge.load_settings(os.path.join(X.ROOT, T.KP_E21_CONFIG)))


def sweep(tmp_path, monkeypatch, ups, skip=()):
    """A made-up sweep of four games x two seeds x three arms. ups[arm][game] = (the level-up actions
    at seed 16, at seed 17). In the explorer's arm every level after the first of a game with keys
    is the explorer's."""
    jobs = [T.Job(g, s, a, 5000) for g in GAMES for s in (16, 17) for a in ARMS]
    monkeypatch.setattr(T, "sweep_jobs", lambda: jobs)
    monkeypatch.setattr(T, "ARM_NAMES", ARMS)
    monkeypatch.setattr(T, "NAME", "THE EXPLORER'S TEST")
    monkeypatch.setattr(V, "ROOT", str(tmp_path))
    rows = []
    for j in jobs:
        if (j.game, j.seed, j.arm) in skip:
            continue
        mine = list(ups[j.arm][j.game][j.seed - 16])
        d = tmp_path / "runs" / j.arm / f"{j.game}_s{j.seed}"
        (d / "kp").mkdir(parents=True)
        explorer = j.arm == "SOT-KP21" and j.game in X.KEYS
        starts = [0] + mine
        levels = [dict(level=k + 1, entered_at=starts[k], left_at=(mine[k] if k < len(mine) else 5000),
                       ended=("not cleared" if k == len(mine) else "cleared by Goose" if k == 0 or not explorer
                              else "cleared by the explorer"),
                       moves=({"goose": 10} if k == 0 else {"explorer": 20}) if explorer else {})
                  for k in range(len(mine) + 1)]
        stats = dict(dormant=False, kp_errors=0, levels=levels,
                     levels_by_planning=max(0, len(mine) - 1) if j.game == "ft09" else 0)
        if j.arm == "SOT-KP21":
            stats.update(explorer="e21", keys=j.game in X.KEYS, levels_by_explorer=max(0, len(mine) - 1) if explorer else 0)
        files = {"metrics.json": {"levels_completed": len(mine), "levelup_events": [[a, k + 1] for k, a in enumerate(mine)]},
                 "run_end.json": {"n_actions": 5000, "elapsed_s": 500.0, "termination": "cap"},
                 "run_config.json": dict(SOT_CFG, kp_config_hash=E21_HASH if j.arm == "SOT-KP21" else T.kp_config_hash()),
                 "train_stats.json": {"train_steps": 4800}, "kp/kp_stats.json": stats}
        for name, content in files.items():
            (d / name).write_text(json.dumps(content))
        rows.append(f"runs/{j.arm}/{j.game}_s{j.seed}\t{j.game}\t{j.seed}\t{j.arm}\n")
    (tmp_path / "manifest.tsv").write_text("".join(rows))
    return str(tmp_path)


# Two runs of one agent: 23 and 21 levels. They differ by a level of tu93 at each seed, by two levels of
# ft09 at one, and in when the levels came.
PLAIN = {"cd82": ([600], [500]), "tu93": ([1500, 1700, 2600, 4000], [1300, 1500, 2700]), "sp80": ([30], [200]),
         "ft09": ([600, 620, 660, 690, 740, 780], [500, 520, 560, 590, 640, 680])}
AGAIN = {"cd82": ([700], [400]), "tu93": ([1600, 1750, 2900], [1200, 1450, 2500, 4300]), "sp80": ([30], [250]),
         "ft09": ([400, 420, 460, 490], [900, 920, 960, 990, 1040, 1080])}
# The explorer: cd82's level 2, and tu93's later levels much sooner; the click-only game as the planner alone.
SOONER = {"cd82": ([650, 1700], [450, 1800]), "tu93": ([1550, 1585, 2150, 2850], [1250, 1280, 1850, 2550]),
          "sp80": ([30], [220]), "ft09": ([500, 520, 560, 590, 640, 680], [700, 720, 760, 790, 840, 880])}


def rules(out):
    runs = V.load(out)
    pairs = sorted(runs["SOT-KP"])
    return X.rule_levels(runs, pairs), X.rule_sooner(runs, pairs)


def test_the_games_with_keys_and_the_games_with_clicks_only_are_the_25():
    assert len(X.KEYS) == 18 and set(X.KEYS) < set(benchmark.ALL_GAMES) and set(X.GROUPMATE) < set(X.KEYS)
    assert set(benchmark.ALL_GAMES) - set(X.KEYS) == {"ft09", "lp85", "r11l", "s5i5", "su15", "tn36", "vc33"}


def test_sooner_but_few_more_levels_clears_rule_b_and_not_rule_a(tmp_path, monkeypatch):
    out = sweep(tmp_path, monkeypatch, {"SOT-KP": PLAIN, "SOT-KP2": AGAIN, "SOT-KP21": SOONER})
    a, b = rules(out)
    assert a["score"] == {"SOT-KP": 23, "SOT-KP21": 26, "SOT-KP2": 21} and a["gap"] == 3 and a["above_both"]
    assert a["bar"] >= 3 and not a["clears"]                    # the two controls come that far apart themselves
    assert b["net"] == 8 and b["net_again"] == 8 and b["pairs"] == 6     # cd82 two races, tu93 six; sp80 has none
    assert b["control"] == 0                                    # the controls: tu93 -1 at one seed, +1 at the other
    assert b["net"] > b["bar"] and b["clears"]
    text = X.write(out)
    assert "**The explorer, more levels: INSIDE THE NOISE at this budget.**" in text
    assert "**The explorer, sooner: CLEARS THE NOISE.**" in text
    assert "| cd82 | dev | yes | 2 | 2 | 4 | 2 | 0 | 80 | 0 | 0 |" in text     # its level 2 at both seeds, by the explorer
    assert "| SOT-KP21 | 10 | 8 | 26 |" in text                 # ft09's ten levels by planning, eight by the explorer
    assert "on the 18 games with keys and on no other: yes" in text and "Each arm with its own settings" in text
    assert os.path.exists(os.path.join(out, "runs.csv")) and os.path.exists(os.path.join(out, "verdict.md"))


def test_many_more_levels_than_two_controls_differ_clears_rule_a(tmp_path, monkeypatch):
    more = dict(SOONER, sp80=([30, 400, 900, 1500], [220, 600, 1300, 2000]))        # three later levels of sp80 as well
    out = sweep(tmp_path, monkeypatch, {"SOT-KP": PLAIN, "SOT-KP2": AGAIN, "SOT-KP21": more})
    a, b = rules(out)
    assert a["gap"] == 9 > a["bar"] and a["clears"] and b["clears"]
    assert "**The explorer, more levels: CLEARS THE NOISE.**" in X.write(out)


def test_a_lead_over_one_control_only_clears_neither_rule(tmp_path, monkeypatch):
    fast = {"cd82": ([700, 1500], [400, 1400]), "tu93": ([1600, 1630, 2100, 2800], [1200, 1230, 1800, 2400]),
            "sp80": ([30, 900], [250]), "ft09": PLAIN["ft09"]}  # a second control that is better than the explorer
    out = sweep(tmp_path, monkeypatch, {"SOT-KP": PLAIN, "SOT-KP2": fast, "SOT-KP21": SOONER})
    a, b = rules(out)
    assert not a["above_both"] and not a["clears"]
    assert b["net"] > 0 >= b["net_again"] and not b["clears"]


def test_level_1_and_the_games_with_clicks_only_are_not_in_rule_b(tmp_path, monkeypatch):
    early = {g: tuple([u[0] - 20] + list(u[1:]) for u in v) for g, v in PLAIN.items()}      # every level 1 a little sooner
    early["ft09"] = ([100, 110, 120, 130, 140, 150], [100, 110, 120, 130, 140, 150])        # and ft09's later levels much sooner
    out = sweep(tmp_path, monkeypatch, {"SOT-KP": PLAIN, "SOT-KP2": PLAIN, "SOT-KP21": early})
    _, b = rules(out)
    runs = V.load(out)
    assert sum(X.sooner(runs, "SOT-KP21", "SOT-KP", [("ft09", 16), ("ft09", 17)]).values()) == 10     # they are races
    assert b["net"] <= 0 and not b["clears"]                    # but not on a game with keys, so not in the rule


def test_no_verdict_before_every_run_is_in(tmp_path, monkeypatch):
    out = sweep(tmp_path, monkeypatch, {"SOT-KP": PLAIN, "SOT-KP2": AGAIN, "SOT-KP21": SOONER},
                skip={("tu93", 17, "SOT-KP2")})
    text = X.write(out)
    assert "No verdict and no table: 1 runs are not in the manifest yet" in text and "CLEARS" not in text
    partial = X.write(out, partial=True)
    assert "NOT GIVEN: the sweep is not complete" in partial and "CLEARS" not in partial
