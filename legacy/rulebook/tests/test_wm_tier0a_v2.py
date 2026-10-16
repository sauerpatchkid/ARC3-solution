"""End-to-end test of Rulebook v2's CP1 run (tools/wm_tier0a_v2.py) on a synthetic
two-level recording, with a fake LLM that answers from a table. No GPU.

The toy game's second level has other colours and another step size, so a rule
only carries if its level facts are in PARAMS and are fitted again.

    uv run python -m pytest tests/test_wm_tier0a_v2.py -q
"""
import argparse
import json
import os
import re
import sys

import numpy as np

TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # legacy/rulebook
ROOT = os.path.dirname(os.path.dirname(TRACK))                        # the repo
sys.path.insert(0, TRACK)                                             # the frozen wm package
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(TRACK, "tools"))
sys.path.insert(0, os.path.join(TRACK, "tests"))

import wm_tier0a_v2 as T  # noqa: E402
from test_wm_cp1 import LEVEL1, LEVEL2, LIGHT, SLIDE, WRONG_WAY, play2  # noqa: E402
from wm import rules2 as R2  # noqa: E402
from wm.llm import ChatClient  # noqa: E402

# v1's style: right on level 1, with the colour in the code.
LITERAL = '''
RULE = "ACTION1 slides the colour-3 block one column right unless it touches the right wall"
def applies(board, act, api):
    return act.action == 1 and (board == 3).any() and np.nonzero(board == 3)[1].max() < 63
def predict(board, act, api):
    out = board.copy()
    ys, xs = np.nonzero(board == 3)
    out[board == 3] = api.background
    out[ys, xs + 1] = 3
    return out
'''
ONE_STEP = SLIDE.replace("Int(1, 3)", "Int(1, 1)")              # clean, but its step cannot be fitted to level 2
TOP = LIGHT.replace("claimed[y0:y0 + 2, x0:x0 + 2] = True", "claimed[y0, x0:x0 + 2] = True") \
    .replace("turns its 2x2 square", "turns the top row of its square")
BOTTOM = LIGHT.replace("claimed[y0:y0 + 2, x0:x0 + 2] = True", "claimed[y0 + 1, x0:x0 + 2] = True") \
    .replace("turns its 2x2 square", "turns the bottom row of its square")
CRASH = SLIDE.replace('xs.max() + p["step"] < 64', "undefined_name")


def block(code):
    return "Thinking is done.\n```python\n" + code.strip() + "\n```"


def write_corpus(d, n=900):
    """Two levels of the toy game as one recorded run, a game over every 40 moves
    (recorded the old way: the action number steps by 3 there)."""
    f1, n1, a1 = play2(n, 0, **LEVEL1)
    f2, n2, a2 = play2(n, 1, **LEVEL2)
    n1[-1] = f2[0]                                             # the level-completing move
    F, N, A = np.concatenate([f1, f2]), np.concatenate([n1, n2]), np.concatenate([a1, a2])
    lv = np.r_[np.zeros(len(a1), np.int32), np.ones(len(a2), np.int32)]
    ends = np.arange(len(A)) % 40 == 39
    nums = 1 + np.arange(len(A)) + 2 * np.cumsum(np.r_[False, ends[:-1]])
    os.makedirs(d)
    for s in range(0, len(A), 300):
        e = slice(s, s + 300)
        np.savez(os.path.join(d, f"shard_{s // 300:05d}.npz"), frames=F[e], next_frames=N[e], actions=A[e],
                 levels=lv[e], action_nums=nums[e].astype(np.int64))


class FakeLLM:
    """Answers by what the prompt asks for, and keeps every prompt it was sent."""

    def __init__(self):
        self.seen = []

    def __call__(self, messages, n):
        last = messages[-1]["content"]
        if isinstance(last, str):                              # a code-only retry or a repair
            self.seen.append(("retry", None, last))
            return [block(LITERAL + "# written on the retry\n")] * n
        text = "\n".join(p["text"] for p in last if p["type"] == "text")
        pictures = sum(p["type"] != "text" for p in last)
        assert pictures <= R2.MAX_IMAGES and all(set(p) <= {"type", "text", "image_url"} for p in last)
        assert messages[0]["content"] == R2.SYSTEM
        group = re.search(r"ACTION GROUP(?: on level \d+)?: (.*?)\.\n", text).group(1)
        kind = next(k for k, marker in (("two-level", R2.BOTH), ("residual", R2.REST), ("deliteral", R2.DELITERAL),
                                        ("refine", R2.IMPROVE), ("retry", R2.AGAIN), ("first", R2.TASK)) if marker in text)
        self.seen.append((kind, group, text))
        if kind == "two-level":
            return [block(SLIDE)]
        if kind == "residual":
            return [block(BOTTOM)]
        if kind == "deliteral":
            return [block(ONE_STEP)]
        if kind == "first" and group == "ACTION1":
            cut = {"text": "thinking and thinking, never reaching code", "finish": "length", "tokens": 20480}
            return ([cut, block(WRONG_WAY), block(CRASH), block(WRONG_WAY)] * n)[:n]
        if kind == "first":
            return ([block(TOP), block(TOP), block(TOP + "# a copy\n"), block(TOP)] * n)[:n]
        raise AssertionError(f"unexpected {kind} prompt for {group}")


def run(tmp_path, llm, name="out", **kw):
    corpus, out = str(tmp_path / "run" / "transitions"), str(tmp_path / name)
    if not os.path.exists(corpus):
        write_corpus(corpus)
    os.makedirs(out, exist_ok=True)
    client = ChatClient(mock=llm, cache_path=os.path.join(out, "cache.jsonl"))
    opts = argparse.Namespace(k=4, calls=8, residual=True, seed=0, manifest="toy", **kw)
    return T.run_game("toy", corpus, out, client, opts), out, client, opts


def test_cp1_run_end_to_end(tmp_path):
    llm = FakeLLM()
    res, out, client, opts = run(tmp_path, llm)
    groups = {g["group"]: g for g in res["groups"]}
    cands = res["candidates"]
    kinds = [(k, g) for k, g, _ in llm.seen]

    # 1. who gets calls: groups whose moves never change the board get none
    assert set(groups) == {"ACTION1", "click on colour 5 (black)"}
    assert "ACTION2" in res["never_changing"] and not any(g == "ACTION2" for _, g in kinds)

    # 2. ACTION1: the truncated answer is retried into v1-style code that is right but level-bound, so
    #    the search does not stop there: the bandit refines it, and the rewrite is clean
    a1 = [c for c in cands if c["group"] == "ACTION1"]
    literal = next(c for c in a1 if c["origin"] == "first round" and c["retried"] and c["stage"] == "checked")
    assert literal["trust"]["trusted"] and literal["lint"] and not literal["has_params"]
    assert ("deliteral", "ACTION1") in kinds
    asked = next(t for k, g, t in llm.seen if k == "deliteral")
    assert "Move each of these into PARAMS" in asked and "board == 3" in asked and "TRUSTED" in asked
    clean = next(c for c in a1 if c["origin"] == "refine")
    assert clean["trust"]["trusted"] and not clean["lint"] and clean["binding"] == {"agent": 3, "step": 1}
    assert groups["ACTION1"]["calls"] == 1 and groups["ACTION1"]["stopped"].startswith("trusted")
    assert sum(c["stage"] == "runtime" for c in a1) == 0       # the crashing answer was repaired once
    assert any(c["retried"] and c["k"] == 2 for c in a1)

    # 3. clicks: the first trusted rule vouches for half the change, so a residual prompt follows
    #    and its answer joins the book
    click = [c for c in cands if c["group"].startswith("click")]
    assert sum(c["duplicate"] for c in click) >= 2
    left = next(t for k, g, t in llm.seen if k == "residual")
    assert "UNEXPLAINED" in left and "turns the top row" in left and "Unexplained changes" in left
    assert groups["click on colour 5 (black)"]["calls"] == 1
    book = [cands[i] for i in res["book"]]
    assert sorted(c["rule"][:40] for c in book) == sorted(r[:40] for r in (
        "ACTION1 slides the agent block right by its step unless",
        "clicking a tile turns the top row of its square to the lit colour",
        "clicking a tile turns the bottom row of its square to the lit colour"))
    assert all(not c["lint"] for c in book)                    # the clean rule is in, its level-bound twin is not
    assert (res["book_train"]["case_coverage"], res["book_train"]["wrong"], res["book_train"]["cell_coverage"]) == (1.0, 0.0, 1.0)

    # 4. the next level. Untouched, the book knows nothing there; re-bound, the click rules carry and
    #    the slide (whose step cannot become 2) is left out; one two-level call repairs it
    tr = res["transfer"]
    assert res["transfer_level"] == 2 and tr["t0"]["per_move"]["rate"]["book"] == tr["t0"]["per_move"]["rate"]["nothing"]
    fate = {c["rule"][:20]: c["t1"] for c in book}
    assert fate["ACTION1 slides the a"]["flag"] == "binding failed" and not fate["ACTION1 slides the a"]["in_t1"]
    assert all(t["in_t1"] and t["binding"] == {"tile": 11, "lit": 12} for r, t in fate.items() if r.startswith("clicking"))
    assert tr["t1"]["rules_in"] == 2 and tr["t1"]["wrong"] == 0.0 and tr["t1"]["flags"]["binding failed"] == 1
    g1 = tr["t1"]["per_move"]
    assert g1["rate"]["nothing"] < g1["rate"]["book"] < 1.0 and g1["diff"]["nothing"]["above_zero"]
    assert ("two-level", "ACTION1") in kinds
    both = next(t for k, g, t in llm.seen if k == "two-level")
    assert "LEVEL 1" in both and "LEVEL 2" in both and "no values of PARAMS make it right" in both
    assert (tr["t2"]["repair_calls"], tr["t2"]["repaired"], tr["t2"]["rules_in"]) == (1, 1, 3)
    g2 = tr["t2"]["per_move"]
    assert g2["rate"]["book"] == 1.0 and g2["beats_both"] and tr["t2"]["case_coverage"] == 1.0
    assert (g2["unit"], g2["units"]) == ("blocks of 50 moves", 12) and g2["moves"] == len_score(res)

    # 5. the cost is counted, and the report and verdict come out
    assert res["llm"]["ran_out"] == 1 and res["llm"]["untried_bindings"] == 0 and res["llm"]["answers"] == len(cands)
    T.report(out, [res], opts, client)
    text = open(os.path.join(out, "report.md")).read()
    assert "**FAIL** (pass = primary and safety)" in text and "T1 beats both baselines" in text
    assert "NOT built (the plan's cut lines): the simplification pass, api.t" in text
    assert "No calls (the board never changed): " in text and "ACTION2 (" in text and "left out: binding failed" in text
    json.load(open(os.path.join(out, "toy.json")))

    # 6. a second pass over the same answers, with new answers refused, gives the same result
    def refuse(messages, n):
        raise AssertionError("an answer was asked for that the first pass did not get")
    again, _, _, _ = run(tmp_path, refuse)
    for k in ("book", "book_train", "transfer"):
        assert json.dumps(again[k], sort_keys=True, default=str) == json.dumps(res[k], sort_keys=True, default=str), k


def len_score(res):
    return res["score"]["moves"]


def test_the_verdict_needs_three_games_and_no_unsafe_one():
    def game(name, beats, wrong, scored=True):
        return {"game": name, "transfer": {"t1": {"per_move": {"beats_both": beats}, "wrong": wrong}} if scored else None}
    three = [game("a", True, 0.0), game("b", True, 0.05), game("c", True, 0.01), game("d", False, 0.0)]
    assert T.verdict(three) == (["a", "b", "c"], [], "PASS")
    assert T.verdict(three[:2] + three[3:]) == (["a", "b"], [], "FAIL")
    assert T.verdict(three + [game("e", False, 0.051)]) == (["a", "b", "c"], ["e"], "FAIL")
    assert T.verdict(three + [game("f", False, 0.0, scored=False)])[1:] == (["f"], "FAIL")   # "on every game"


def test_without_residual_prompts_a_group_stops_at_its_first_carrying_rule(tmp_path):
    llm = FakeLLM()
    corpus, out = str(tmp_path / "run" / "transitions"), str(tmp_path / "plain")
    write_corpus(corpus)
    os.makedirs(out)
    client = ChatClient(mock=llm, cache_path=os.path.join(out, "cache.jsonl"))
    opts = argparse.Namespace(k=4, calls=8, residual=False, seed=0, manifest="toy")
    res = T.run_game("toy", corpus, out, client, opts)
    groups = {g["group"]: g for g in res["groups"]}
    assert groups["click on colour 5 (black)"]["calls"] == 0 and groups["click on colour 5 (black)"]["stopped"] == "first trusted rule"
    assert not any(k == "residual" for k, _, _ in llm.seen)
    assert 0 < res["book_train"]["cell_coverage"] < 1.0        # half of each click is left unexplained


def test_all_games_share_one_pool_and_a_failed_game_does_not_stop_the_others(tmp_path):
    corpus = str(tmp_path / "run" / "transitions")
    write_corpus(corpus)
    alone, _, _, _ = run(tmp_path, FakeLLM(), name="alone")
    out = str(tmp_path / "pooled")
    os.makedirs(out)
    client = ChatClient(mock=FakeLLM(), cache_path=os.path.join(out, "cache.jsonl"))
    opts = argparse.Namespace(k=4, calls=8, residual=True, seed=0, manifest="toy")
    runs = {"toy": corpus, "copy": corpus, "lost": str(tmp_path / "nowhere")}
    results, failed = T.run_all(["toy", "lost", "copy"], runs, out, client, opts)
    assert failed == ["lost"] and [r["game"] for r in results] == ["toy", "copy"]
    for r in results:                                          # the same answers give the same result as one game alone
        for k in ("book", "book_train", "transfer"):
            assert json.dumps(r[k], sort_keys=True, default=str) == json.dumps(alone[k], sort_keys=True, default=str), k
    os.makedirs(tmp_path / "evidence")
    empty, failed = T.run_all(["toy"], runs, str(tmp_path / "evidence"), None, opts)   # --evidence-only
    assert not failed and empty[0]["book"] == [] and empty[0]["transfer"]["t1"]["rules_in"] == 0
