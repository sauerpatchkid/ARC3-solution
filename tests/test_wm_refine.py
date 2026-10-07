"""The near-miss refinement bandit (tools/wm_refine.py), end to end on the toy
game with a mock LLM: no server, no GPU."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import wm_refine  # noqa: E402
from test_wm_core import GOOD_MOVE, OFF_BY_ONE, play  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.llm import ChatClient  # noqa: E402

# Right except when the block's right edge sits on a column divisible by 10: a near miss.
NEAR = GOOD_MOVE.replace("    out = board.copy()\n    ys, xs = np.nonzero(board == 3)\n",
                         "    out = board.copy()\n    ys, xs = np.nonzero(board == 3)\n"
                         "    if xs.max() % 10 == 0:\n        return out\n")


def block(code):
    return "Fixed.\n```python\n" + code.strip() + "\n```"


def v1_folder(tmp_path, monkeypatch):
    f, n, a = play(900)
    LevelEvidence.from_moves("toy", 0, f, n, a).save(str(tmp_path / "toy_train.npz"))
    f2, n2, a2 = play(400, seed=1, shift=1)
    LevelEvidence.from_moves("toy", 1, f2, n2, a2).save(str(tmp_path / "toy_transfer.npz"))
    cands = [{"group": "ACTION1", "code": c} for c in (NEAR, OFF_BY_ONE, NEAR)] + [{"group": "ACTION1", "code": None}]
    (tmp_path / "toy.json").write_text(json.dumps({"candidates": cands}))
    monkeypatch.setattr(wm_refine, "V1", str(tmp_path))


def test_starting_pool_keeps_only_usable_rules(tmp_path, monkeypatch):
    v1_folder(tmp_path, monkeypatch)
    ev = LevelEvidence.load(str(tmp_path / "toy_train.npz"))
    pool, n = wm_refine.starting_pool("toy", "ACTION1", ev, str(tmp_path / "toy_train.npz"))
    assert n == 2 and len(pool) == 1                       # duplicates merged; the 0%-right rule is no arm
    arm = pool[0]
    assert 0.5 < arm["grade"]["accuracy"] < 1 and not wm_refine.is_exact(arm)
    parts, shown = wm_refine.refine_parts(ev, "ACTION1", arm, wm_refine.rng_for("p"), wm_refine.rng_for("c"))
    assert 1 <= shown <= wm_refine.N_FAILING
    assert sum(p["type"] == "image_url" for p in parts) <= 10      # the server's per-prompt limit
    text = "\n".join(p["text"] for p in parts if p["type"] == "text")
    assert "The rule was WRONG on this move" in text and "```python" in text


def test_the_bandit_refines_until_exact_and_then_stops(tmp_path, monkeypatch):
    v1_folder(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    answers = iter([{"text": "thinking forever", "finish": "length", "tokens": 20480},   # runs out of room...
                    block(NEAR),                                                       # ...its retry: no better
                    block(GOOD_MOVE)])                                                 # second call: exact
    client = ChatClient(mock=lambda m, n: [next(answers)], cache_path=str(out / "cache.jsonl"))
    res = wm_refine.run_group("toy", "ACTION1", client, str(out), calls=5)
    assert res["reached_exact"] and res["exact_at_step"] == 2 and res["calls"] == 2
    assert res["steps"][0]["retried"] and res["steps"][0]["duplicate"]       # same code as its parent
    best = res["best"]
    assert best["grade"]["plan_eligible"] and best["group_changing_covered"] == 1.0
    assert best["transfer"]["right"] == best["transfer"]["cases"] > 0        # carries to the shifted level
    saved = json.load(open(out / "toy__ACTION1.json"))
    assert saved["reached_exact"] and saved["starting_best_accuracy"] < 1
    wm_refine.report(str(out), [res], client)
    assert "1 of 1 groups reached an exact rule" in open(out / "report.md").read()


def test_thompson_prefers_accurate_rules_not_yet_refined():
    arm = lambda acc, refined: {"grade": {"accuracy": acc}, "refined": refined}
    rng = np.random.default_rng(0)
    picks = [wm_refine.pick([arm(0.95, 0), arm(0.60, 0), arm(0.95, 30)], rng) for _ in range(400)]
    counts = np.bincount(picks, minlength=3)
    assert counts[0] > counts[1] and counts[0] > counts[2] and counts.min() >= 0
    assert wm_refine.verdict(3).startswith("3 or more") and wm_refine.verdict(2).startswith("Exactly 2")
    assert wm_refine.verdict(1).startswith("1 or fewer")
