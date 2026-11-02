"""tools/gb_gate.py: G3's verdict (pre-registration section 6) on two fake runs."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gb_gate  # noqa: E402
import tiers  # noqa: E402


def fake_run(root, name, games, coverage, trusted, wrong, ran_out, answers, templates=0):
    """A run folder holding only what gb_gate reads, the same numbers on every game unless given per game."""
    d = os.path.join(root, name)
    for g in games:
        pick = lambda x: x[g] if isinstance(x, dict) else x
        os.makedirs(os.path.join(d, "s3_check", g), exist_ok=True)
        os.makedirs(os.path.join(d, "s2_write", g), exist_ok=True)
        t1 = {"per_move": {"rate": {"book": 0.5, "nothing": 0.1, "memory": 0.6}, "diff": {"nothing": 0.4, "memory": -0.1},
                           "beats_both": False}, "wrong": pick(wrong), "members": [0], "changing_cases": 1000}
        guards = {"book": {"after": {"cell_coverage": pick(coverage), "case_coverage": 0.1, "members": [0], "wrong": 0.0}},
                  "t1": {"after": t1}, "rescued": []}
        with open(os.path.join(d, "s3_check", f"{g}_L1.guards.json"), "w") as f:
            json.dump(guards, f)
        with open(os.path.join(d, "s3_check", g, "rules.jsonl"), "w") as f:
            for i in range(pick(trusted)):
                f.write(json.dumps({"v2_trusted": True, "v2_admitted": True}) + "\n")
        cands = [{"id": i, "stage": "checked", "origin": "first round", "trust": {"trusted": False},
                  "finished": i >= pick(ran_out)} for i in range(answers)]
        cands += [{"id": answers + i, "stage": "checked", "origin": "template", "template": i, "trust": {"trusted": True}}
                  for i in range(templates)]
        with open(os.path.join(d, "s2_write", g, "candidates.jsonl"), "w") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
        with open(os.path.join(d, "s2_write", g, "llm.json"), "w") as f:
            json.dump({"calls": 3, "candidates": len(cands), "tokens": 10, "ran_out": pick(ran_out), "groups": [],
                       "requests": 5 - min(templates, 2)}, f)
    return type("R", (), {"name": name, "dir": d})


def test_g3_needs_more_rules_and_coverage_on_five_games_and_no_more_wrong_or_overruns(tmp_path):
    games = list(tiers.TIERS["ab_offline"])
    up = {g: (0.3 if i < 5 else 0.2) for i, g in enumerate(games)}             # higher on 5 of 8
    more = {g: (3 if i < 5 else 2) for i, g in enumerate(games)}
    gb = fake_run(str(tmp_path), "gb", games, 0.2, 2, 0.01, 2, 10)
    gbt = fake_run(str(tmp_path), "gbt", games, up, more, 0.01, 2, 10, templates=1)
    v = gb_gate.g3(gbt, gb)
    assert v["better"] == games[:5] and v["primary"] and v["wrong_ok"] and v["overrun_ok"] and v["pass"]
    assert v["wrong"]["gbt"] == {"count": 80, "of": 8000, "share": 0.01}
    assert v["requests_saved"][games[0]] == 1 and v["negative_transfer"] == []
    worse = fake_run(str(tmp_path), "gbt2", games, up, more, 0.02, 3, 10, templates=1)
    v = gb_gate.g3(worse, gb)
    assert v["primary"] and not v["wrong_ok"] and not v["overrun_ok"] and not v["pass"]
    four = fake_run(str(tmp_path), "gbt3", games, {g: (0.3 if i < 4 else 0.2) for i, g in enumerate(games)}, more,
                    0.01, 2, 10)
    assert not gb_gate.g3(four, gb)["primary"]
