"""Unit tests for the scoring fixes: compute_metrics.apply_run_end (engine level
count from run_end.json) and tools/paired_compare.py (--expect, bootstrap).

    uv run python -m pytest tests/ -q
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from compute_metrics import apply_run_end, compute  # noqa: E402
from paired_compare import bootstrap_delta, expect_check, load, rule_one  # noqa: E402


# ---- compute_metrics: run_end.json ------------------------------------------------
def _corpus(tmp_path, levels):
    """A tiny run folder: transitions/shard_00000.npz with one transition per
    entry in `levels` (action_num 1..n), every frame different."""
    run = tmp_path / "run"
    (run / "transitions").mkdir(parents=True)
    n = len(levels)
    frames = np.zeros((n, 64, 64), np.uint8)
    frames[np.arange(n), 0, np.arange(n)] = 1
    np.savez(run / "transitions" / "shard_00000.npz",
             frames=frames, next_frames=np.roll(frames, -1, axis=0),
             actions=np.zeros(n, np.int32), changed=np.ones(n, np.uint8),
             levels=np.array(levels, np.int32), action_nums=np.arange(1, n + 1, dtype=np.int64),
             wall_ms=np.ones(n, np.float32), model_ms=np.ones(n, np.float32))
    return run


def _end(run, levels, n_actions, termination="cap"):
    (run / "run_end.json").write_text(json.dumps(
        {"termination": termination, "final_levels": levels, "n_actions": n_actions}))


def test_no_run_end_scores_exactly_as_before(tmp_path):
    run = _corpus(tmp_path, [0, 0, 1, 1])
    m = compute(str(run / "transitions"))
    before = dict(m)
    m = apply_run_end(m, str(run / "transitions"))
    assert m == before and m["levels_completed"] == 1 and "termination" not in m


def test_agreeing_engine_count_only_adds_fields(tmp_path):
    run = _corpus(tmp_path, [0, 0, 1, 1])
    _end(run, 1, 5)
    m = apply_run_end(compute(str(run / "transitions")), str(run / "transitions") + "/")
    assert m["levels_completed"] == 1 and m["levels_engine"] == 1
    assert m["termination"] == "cap" and m["levelup_events"] == [(3, 1)]


def test_final_move_level_up_is_added_at_the_last_action(tmp_path):
    run = _corpus(tmp_path, [0, 0, 1, 1])
    _end(run, 2, 5, termination="win")
    m = apply_run_end(compute(str(run / "transitions")), str(run / "transitions"))
    assert m["levels_completed"] == 2 and m["levels_completed_corpus"] == 1
    assert m["levelup_events"][-1] == (5, 2) and m["max_level"] == 2


def test_final_move_level_up_with_no_earlier_level(tmp_path):
    run = _corpus(tmp_path, [0, 0, 0])
    _end(run, 1, 4)
    m = apply_run_end(compute(str(run / "transitions")), str(run / "transitions"))
    assert m["levels_completed"] == 1 and m["first_levelup_action"] == 4


def test_unexplained_gap_uses_engine_count_without_inventing_events(tmp_path):
    run = _corpus(tmp_path, [0, 0, 1, 1])
    _end(run, 4, 5)
    m = apply_run_end(compute(str(run / "transitions")), str(run / "transitions"))
    assert m["levels_completed"] == 4 and m["levelup_events"] == [(3, 1)]


# ---- paired_compare: --expect and bootstrap ---------------------------------------
def _manifest(tmp_path, name, rows, cap=100000, end=None):
    """rows: [(game, seed, levels)] -> a manifest file with one run dir each."""
    lines = []
    for i, (g, s, lv) in enumerate(rows):
        d = tmp_path / f"{name}_{i}"
        d.mkdir()
        m = {"levels_completed": lv, "unique_states": 10, "n_actions": 100,
             "novelty_late_per_1k": 1.0, "actions_per_sec": 100.0}
        if end:
            m["termination"] = end
        (d / "metrics.json").write_text(json.dumps(m))
        (d / "run_config.json").write_text(json.dumps({"max_actions": cap}))
        lines.append(f"{d}\t{g}\t{s}\tX")
    path = tmp_path / f"{name}.tsv"
    path.write_text("\n".join(lines) + "\n")
    return f"{path}:X"


ROWS = [("g1", "0", 1), ("g1", "1", 1), ("g2", "0", 0), ("g2", "1", 2)]


def test_expect_passes_on_a_complete_matching_set(tmp_path):
    probs = []
    b, n = load(_manifest(tmp_path, "b", ROWS), probs), load(_manifest(tmp_path, "n", ROWS), probs)
    assert expect_check(4, b, n, sorted(set(b) & set(n)), probs) == []


def test_expect_blocks_missing_duplicate_cap_and_error_runs(tmp_path):
    probs = []
    b = load(_manifest(tmp_path, "b", ROWS + [("g1", "0", 3)]), probs)       # duplicate
    n = load(_manifest(tmp_path, "n", ROWS[:3], cap=50000, end="error"), probs)  # missing, cap, error
    bad = " | ".join(expect_check(4, b, n, sorted(set(b) & set(n)), probs))
    assert "duplicate run for g1 seed 0" in bad
    assert "new has 3 runs" in bad and "3 (game, seed) pairs" in bad
    assert "action caps differ" in bad and "ended with 'error'" in bad


def test_historical_runs_without_termination_are_accepted(tmp_path):
    probs = []
    b = load(_manifest(tmp_path, "b", ROWS), probs)
    n = load(_manifest(tmp_path, "n", ROWS, end="win"), probs)
    assert expect_check(4, b, n, sorted(set(b) & set(n)), probs) == []


def test_bootstrap_is_deterministic_and_brackets_a_constant_delta():
    assert bootstrap_delta([2, 2, 2]) == (6, 6)
    assert bootstrap_delta([0, 5, -1, 3]) == bootstrap_delta([0, 5, -1, 3])
    lo, hi = bootstrap_delta([0, 5, -1, 3])
    assert lo <= 7 <= hi


def test_the_three_verdict_rules_differ_only_on_ties():
    assert all(rule_one(r, 113, 112, 5, 3) for r in ("adopt", "dev", "confirm"))
    # equal levels: the confirm rule needs strictly more
    assert rule_one("adopt", 112, 112, 5, 3) and rule_one("dev", 112, 112, 5, 3)
    assert not rule_one("confirm", 112, 112, 5, 3)
    # as many losses as wins: only the dev rule accepts
    assert rule_one("dev", 113, 112, 4, 4)
    assert not rule_one("adopt", 113, 112, 4, 4) and not rule_one("confirm", 113, 112, 4, 4)
    assert not any(rule_one(r, 111, 112, 9, 0) for r in ("adopt", "dev", "confirm"))
