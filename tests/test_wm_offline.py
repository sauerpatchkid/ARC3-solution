"""End-to-end test of the Rulebook's offline rule test (tools/wm_offline.py) on a
synthetic recorded run, with a fake LLM that answers from a table. No GPU.

    uv run python -m pytest tests/ -q
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import wm_offline  # noqa: E402
from test_wm_core import GOOD_CLICK, GOOD_MOVE, NOOP, OFF_BY_ONE, play  # noqa: E402
from wm import rules as R  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.llm import ChatClient  # noqa: E402


def block(code):
    return "The block moves.\n```python\n" + code.strip() + "\n```"


def click_noop(colour):
    return (f'RULE = "clicking colour {colour} does nothing"\n'
            f"def applies(board, act, api):\n    return act.action == 6 and board[act.click] == {colour}\n"
            "def predict(board, act, api):\n    return board.copy()\n")


def fake_llm(messages, n):
    """Answers by action group. ACTION1's first answer is truncated (no code), so
    the code-only retry is exercised; its second is wrong, its third right."""
    last = messages[-1]["content"]
    if isinstance(last, str):                                  # a retry: FINISH or REPAIR
        return [block(GOOD_MOVE + "# written on the retry\n")] * n
    text = "\n".join(p["text"] for p in last if p["type"] == "text")
    group = re.search(r"ACTION GROUP: (.*?)\.\n", text).group(1)
    if group == "ACTION1":
        cut = {"text": "thinking and thinking, never reaching code", "finish": "length", "tokens": 20480}
        return ([cut, block(OFF_BY_ONE), block(GOOD_MOVE)] * n)[:n]
    if group == "ACTION2":
        return [block(NOOP)] * n
    colour = int(re.search(r"click on colour (\d+)", group).group(1))
    return [block(GOOD_CLICK if colour == 5 else click_noop(colour))] * n


def write_corpus(d):
    """Two levels of the toy game as one recorded run (shards of 300 moves)."""
    f1, n1, a1 = play(700, seed=0)
    f2, n2, a2 = play(400, seed=1, shift=1)
    n1[-1] = f2[0]                                             # the level-completing move
    F, N, A = np.concatenate([f1, f2]), np.concatenate([n1, n2]), np.concatenate([a1, a2])
    lv = np.r_[np.zeros(700, np.int32), np.ones(400, np.int32)]
    os.makedirs(d)
    for s in range(0, len(A), 300):
        e = slice(s, s + 300)
        np.savez(os.path.join(d, f"shard_{s // 300:05d}.npz"), frames=F[e], next_frames=N[e], actions=A[e],
                 levels=lv[e], action_nums=np.arange(1, len(A) + 1, dtype=np.int64)[e],
                 changed=(F[e] != N[e]).reshape(len(A[e]), -1).any(axis=1).astype(np.uint8))


def test_offline_rule_test_end_to_end(tmp_path):
    corpus, out = str(tmp_path / "run" / "transitions"), str(tmp_path / "out")
    write_corpus(corpus)
    os.makedirs(out)
    client = ChatClient(mock=fake_llm, cache_path=os.path.join(out, "cache.jsonl"))
    a = argparse.Namespace(k=3, rounds=2, manifest="toy")
    res = wm_offline.run_game("toy", corpus, out, client, a)

    assert res["level1"]["level"] == 1 and res["level1"]["moves"] == 699      # the winning move is left out
    assert res["level2"]["level"] == 2
    rules = {c["rule"] for c in res["book"]}
    assert any("slides the colour-3 block" in r for r in rules) and "ACTION2 does nothing" in rules
    move = next(c for c in res["book"] if "slides" in c["rule"])
    assert move["grade"]["plan_eligible"] and move["transfer"]["right"] == move["transfer"]["keys"] > 0
    cov = res["coverage"]
    assert cov["wrong"] == 0 and cov["coverage"] > 0.5
    t = res["transfer"]
    assert t["book"] > t["nothing"] and t["test_wrong"] == 0
    stages = res["rounds"][0]["stages"]
    assert stages.get("no-code", 0) >= 1 and stages.get("duplicate", 0) >= 1
    assert any(c.get("repaired") for c in res["candidates"])               # the code-only retry produced a rule
    assert json.load(open(os.path.join(out, "toy.json")))["game"] == "toy"
    wm_offline.report(out, [res], a, client)
    assert "G1" in open(os.path.join(out, "report.md")).read()

    # a second run answers entirely from the cache
    c2 = ChatClient(mock=lambda m, n: 1 / 0, cache_path=os.path.join(out, "cache.jsonl"))
    res2 = wm_offline.run_game("toy", corpus, out, c2, a)
    assert res2["coverage"] == res["coverage"] and c2.meter["cached"] == c2.meter["requests"]


def test_prompt_parts_show_distinct_effects_and_pictures(tmp_path):
    f, n, a = play(700, seed=0)
    ev = LevelEvidence.from_moves("toy", 0, f, n, a)
    parts = R.group_prompt(ev, "ACTION1", np.random.default_rng(0))
    texts = [p["text"] for p in parts if p["type"] == "text"]
    assert sum(p["type"] == "image_url" for p in parts) == 1 + sum(t.startswith("Move ") for t in texts)
    assert any("colour-3 shape moved by (+0 row," in t for t in texts)
    assert any("nothing changed" in t for t in texts)                       # the wall case is shown too
    conv = R.conversation(parts)
    assert conv[0]["role"] == "system" and "def applies(board, act, api)" in conv[0]["content"]
    assert R.extract_code("thoughts</think>\n```python\nRULE = 'x'\n```") == "RULE = 'x'"
    assert R.extract_code("no code here") is None
    # a truncated answer is all thinking: the retry must still see that reasoning
    assert R.tail("the block moves 5 rows</think>") == "the block moves 5 rows"
    assert R.tail("plain thinking, never closed") == "plain thinking, never closed"
    assert R.tail("thoughts</think>the answer") == "the answer"
    assert R.tail("x" * 30000 + "</think>").startswith("[...earlier reasoning omitted...]")


def test_a_dropped_connection_is_retried_and_a_rejected_request_is_not(monkeypatch):
    import io
    import urllib.error
    import wm.llm as L
    reply = json.dumps({"choices": [{"index": 0, "message": {"content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 3}}).encode()
    calls = {"n": 0}

    class Reply(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionResetError("dropped")
        return Reply(reply)
    monkeypatch.setattr(L.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    c = ChatClient(retries=3)
    c._served = "m"
    assert c.complete([{"role": "user", "content": "x"}])[0]["text"] == "ok"
    assert calls["n"] == 3 and c.meter["retries"] == 2

    def rejected(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 400, "bad request", {}, None)
    calls["n"] = 0
    monkeypatch.setattr(L.urllib.request, "urlopen", rejected)
    try:
        c.complete([{"role": "user", "content": "y"}])
        assert False, "a 400 must be raised"
    except urllib.error.HTTPError:
        pass
    assert calls["n"] == 1                       # not retried
