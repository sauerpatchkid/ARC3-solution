"""S6 SHOW (custom_agents/gb/s6_show.py) on a tiny run built from the toy game:
the page builds offline, its numbers come from the run's files, every case it
draws re-runs to the status the stage stored, and a stored status that the rules
do not reproduce stops the build."""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from gb import artifacts as A  # noqa: E402
from gb import s6_show  # noqa: E402
from gb.core import bind as B  # noqa: E402
from gb.core import book as K  # noqa: E402
from gb.core.check import EXACT, UNKNOWN, WRONG  # noqa: E402
from gb.core.evidence import LevelEvidence  # noqa: E402
from test_gb_core import GOOD_CLICK, OFF_BY_ONE, play  # noqa: E402

RULES = {0: ("click on colour 5 (black)", GOOD_CLICK), 1: ("ACTION1", OFF_BY_ONE)}   # id: (group, code)


def make_run(root, name="toy_run"):
    """A run folder laid out as S3 and S4 write it, for one toy game."""
    run = A.Run(name, root=str(root))
    ev_dir = os.path.join(run.dir, "evidence")
    os.makedirs(ev_dir)
    f, n, a = play(600, seed=0)
    train = LevelEvidence.from_moves("toy", 0, f, n, a)
    p_train = os.path.join(ev_dir, "toy_L1_train.npz")
    train.save(p_train)
    f2, n2, a2 = play(400, seed=1, shift=1)
    nxt = LevelEvidence.from_moves("toy", 1, f2, n2, a2, mask=train.mask)
    p_fit, p_score = os.path.join(ev_dir, "toy_L2_fit.npz"), os.path.join(ev_dir, "toy_L2_score.npz")
    nxt.save(p_fit)
    nxt.save(p_score)
    s2 = os.path.join(run.stage_dir("s2_write"), "toy")
    os.makedirs(s2)
    with open(os.path.join(s2, "candidates.jsonl"), "w") as fh:
        for i, (g, code) in RULES.items():
            fh.write(json.dumps({"id": i, "group": g, "code": code}) + "\n")
    s3 = run.stage_dir("s3_check")
    os.makedirs(os.path.join(s3, "toy"))
    res = {i: B.run_rule(code, p_train) for i, (g, code) in RULES.items()}
    with open(os.path.join(s3, "toy", "rules.jsonl"), "w") as fh:
        for i, (g, code) in RULES.items():
            fh.write(json.dumps({"id": i, "group": g, "binding": {}, "rule": code.split('"')[1],
                                 "origin": "first round", "v2_admitted": True, "v2_trusted": True}) + "\n")
    np.savez_compressed(os.path.join(s3, "toy", "rule_cases.npz"), ids=np.array(list(RULES)),
                        applies=np.array([res[i]["applies"] for i in RULES]),
                        claimed_ok=np.array([res[i]["claimed_ok"] for i in RULES]),
                        cells_right=np.array([res[i]["cells_right"] for i in RULES]))
    members = [dict(src=code, binding={}, layout_path=p_train, group=g) for g, code in RULES.values()]
    comp = K.compose(members, p_train)
    with open(os.path.join(s3, "toy_L1.book.json"), "w") as fh:
        json.dump({"members": list(RULES), "groups_called": [g for g, _ in RULES.values()],
                   "numbers": K.numbers(comp, train)}, fh)
    keys = ("status", "hash", "n_rules", "cells_right", "cells_wrong", "cells_clash", "claimed_n")
    np.savez_compressed(os.path.join(s3, "toy_L1.status.npz"), members=np.array(list(RULES)),
                        **{k: np.asarray(comp[k]) for k in keys})
    s4 = run.stage_dir("s4_carry")
    comp1 = K.compose([dict(m, layout_path=p_fit) for m in members], p_score)
    nums = K.numbers(comp1, nxt)
    per_move = {"rate": {"book": 0.5, "nothing": 0.4, "memory": 0.45}, "moves": 400, "unit": "attempts", "units": 3,
                "diff": {b: {"point": 0.1, "lo": None, "hi": None} for b in ("nothing", "memory")},
                "beats_both": False}
    with open(os.path.join(s4, "toy_L2.carry.json"), "w") as fh:
        json.dump({"rules": [{"id": i, "in_t1": True, "binding": {}, "text": "re-bound"} for i in RULES],
                   "t1": dict(nums, per_move=per_move, rules_in=2)}, fh)
    np.savez_compressed(os.path.join(s4, "toy_L2.t1_status.npz"), members=np.array(list(RULES)),
                        **{k: np.asarray(comp1[k]) for k in keys})
    return run, train, comp


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("glassbox")
    run, train, comp = make_run(root)
    path = s6_show.build(run.name, root=str(root), log=lambda m: None)
    return root, run, train, comp, path


def page_data(path):
    html = open(path).read()
    m = re.search(r'<script id="glassbox-data" type="application/json">(.*?)</script>', html, re.S)
    return html, json.loads(m.group(1))


def test_the_page_is_self_contained(built):
    html, data = page_data(built[4])
    assert not re.search(r'(src|href)="https?:|<link|@import|url\(http', html)   # nothing fetched from anywhere
    assert data["meta"]["run"] == "toy_run" and set(data["games"]) == {"toy"}
    assert os.path.exists(os.path.join(os.path.dirname(built[4]), "figures", "funnel.png"))


def test_the_funnel_counts_come_from_the_run(built):
    _, run, train, comp, path = built
    f = page_data(path)[1]["funnel"][0]
    assert f["changing"] == int(train.changed.sum()) and f["in_called"] <= f["changing"]
    assert f["book_whole"] == int((train.changed & (comp["status"] == EXACT)).sum())
    assert f["book_part"] >= f["book_whole"] and f["transfer"]["level"] == 2


def test_the_viewer_shows_flagged_cases_with_their_rules(built):
    cases = json.load(open(os.path.join(os.path.dirname(built[4]), "cases", "toy.json")))
    train = cases["levels"][0]
    flags = {fl for c in train["cases"] for fl in c["flags"]}
    assert {"WRONG", "EXACT"} <= flags                     # the off-by-one move rule is wrong; clicks are exact
    wrong = next(c for c in train["cases"] if "WRONG" in c["flags"])
    assert wrong["rules"] == [1] and wrong["status"] == "WRONG" and wrong["cells"]["wrong"] > 0
    assert len(cases["levels"]) == 2 and set(cases["rules"]) == {"0", "1"}


def test_a_stored_status_the_rules_do_not_reproduce_stops_the_build(built):
    root, run = built[0], built[1]
    cases = json.load(open(os.path.join(os.path.dirname(built[4]), "cases", "toy.json")))
    k = next(c["k"] for c in cases["levels"][0]["cases"] if c["status"] == "WRONG")   # a case the viewer draws
    p = os.path.join(run.dir, "s3_check", "toy_L1.status.npz")
    st = dict(np.load(p))
    assert st["status"][k] == WRONG
    st["status"][k] = UNKNOWN                               # a stage that wrote the wrong status
    np.savez_compressed(p, **st)
    with pytest.raises(RuntimeError, match="re-run gives"):
        s6_show.build(run.name, root=str(root), log=lambda m: None)
