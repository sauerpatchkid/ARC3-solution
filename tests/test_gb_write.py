"""S2 WRITE (custom_agents/gb/s2_write.py) with a mock LLM on the synthetic maze: what
the prompts hold, how answers become graded candidates, the retry and repair,
the guard step that saves a call, and that S3 and S4 take S2's files."""
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
from gb import h1_vocab as H  # noqa: E402
from gb import s1_see, s3_check, s4_carry  # noqa: E402
from gb import s2_write as W  # noqa: E402
from gb.core.evidence import LevelEvidence  # noqa: E402
from gb.core.llm import ChatClient  # noqa: E402
from test_gb_guards import ZERO, maze, move_rule  # noqa: E402

NEAR = move_rule(3, 0, -1, False)                      # ACTION3: wrong where the mover is blocked
EXACT = move_rule(1, -1, 0, True)                      # ACTION1: checks for itself
BROKEN = "RULE = 'oops'\nPARAMS = {}\ndef applies(board, act, api, p):\n    return undefined_name\n" \
         "def predict(board, act, api, p):\n    return board\n"


def block(code):
    return f"Here it is.\n```python\n{code}\n```"


def group_of(messages):
    texts = [p["text"] for p in messages[1]["content"] if p["type"] == "text"]
    return re.search(r"ACTION GROUP: (\S+)\.", "\n".join(texts)).group(1)


def mock(log):
    """First round: ACTION3 gets the near miss, an answer that ran out of room, and a broken rule;
    the others get the exact rule shifted to their own action. Retries and repairs get the rule."""
    def answer(messages, n):
        group = group_of(messages)
        log.append((group, len(messages), n))
        rule = NEAR if group == "ACTION3" else EXACT.replace("act.action != 1", f"act.action != {group[-1]}")
        if group == "ACTION4":
            rule = move_rule(4, 0, 1, True)
        elif group == "ACTION2":
            rule = move_rule(2, 1, 0, True)
        if len(messages) > 2:                          # a code-only retry or a repair
            return [block(rule)]
        if group == "ACTION3" and n == 4:
            return [block(NEAR), {"text": "thinking...", "finish": "length", "tokens": 20480}, block(BROKEN),
                    block(NEAR)]
        return [block(rule)] * n
    return answer


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    run = A.Run("write", root=str(tmp_path_factory.mktemp("glassbox")))
    ev_dir = os.path.join(run.dir, "evidence")
    os.makedirs(ev_dir)
    f, n, a, o = maze(0, 9, 2000)
    LevelEvidence.from_moves("maze", 0, f, n, a, mask=ZERO, order=o).save(os.path.join(ev_dir, "maze_L0_train.npz"))
    f, n, a, o = maze(1, 12, 1500)
    part = lambda s: {k: v[s] for k, v in o.items()}
    LevelEvidence.from_moves("maze", 2, f[:300], n[:300], a[:300], mask=ZERO, order=part(slice(None, 300))).save(
        os.path.join(ev_dir, "maze_L2_fit.npz"))
    LevelEvidence.from_moves("maze", 2, f[300:], n[300:], a[300:], mask=ZERO, order=part(slice(300, None))).save(
        os.path.join(ev_dir, "maze_L2_score.npz"))
    s1_see.run_game(run, "maze", log=lambda m: None)
    calls = []
    client = ChatClient(mock=mock(calls), cache_path=os.path.join(run.stage_dir("s2_write"), "llm_cache.jsonl"))
    lines = []
    out = W.run_all(run, ["maze"], client, log=lines.append)
    return run, out["maze"], calls, lines


def cands(run):
    with open(os.path.join(run.dir, "s2_write", "maze", "candidates.jsonl")) as f:
        return [json.loads(line) for line in f]


def test_the_prompt_holds_the_facts_and_the_vocabulary(written):
    run, *_ = written
    p_train, ev, texts, *_ = W.level_inputs(run, "maze")
    prompts = W.Prompter(ev, texts, W.DEFAULTS)
    parts, instruction = prompts.first_round("ACTION3", W.rng_for("maze", "ACTION3", "prompt"))
    text = "\n".join(p["text"] for p in parts if p["type"] == "text")
    assert "Seen as objects:" in text and "The mover (colour 9 blue" in text
    assert H.CARD in prompts.system and "ONE MECHANISM PER RULE" in prompts.system
    assert sum(p["type"] != "text" for p in parts) <= W.MAX_IMAGES
    assert prompts.estimate_tokens(parts) <= W.MAX_PROMPT_TOKENS and "ONE mechanism" in instruction
    bare = W.Prompter(ev, texts, dict(W.DEFAULTS, facts=False), vocab=False)
    parts, _ = bare.first_round("ACTION3", W.rng_for("maze", "ACTION3", "prompt"))
    assert "Seen as objects" not in "\n".join(p["text"] for p in parts if p["type"] == "text")
    assert H.CARD not in bare.system


def test_the_groups_called_and_their_order(written):
    run, *_ = written
    groups = json.load(open(os.path.join(run.dir, "s2_write", "maze", "groups.json")))
    assert sorted(groups) == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def test_exact_rules_stop_their_group_without_a_call(written):
    run, llm, calls, _ = written
    by_group = {g["group"]: g for g in llm["groups"]}
    for g in ("ACTION1", "ACTION2", "ACTION4"):
        assert by_group[g]["calls"] == 0 and by_group[g]["carrying"] >= 1, by_group[g]


def test_a_near_miss_is_fixed_by_the_guard_with_no_call_spent(written):
    run, llm, calls, lines = written
    g3 = next(g for g in llm["groups"] if g["group"] == "ACTION3")
    assert g3["guards"] == 1 and g3["calls"] == 0 and g3["carrying"] == 1
    guard = [c for c in cands(run) if c["origin"] == "guard"]
    assert len(guard) == 1 and guard[0]["trust"]["trusted"]
    assert guard[0]["guard"]["words"] == "the mover is touching something on its left"
    assert any("guard on rule" in line for line in lines)


def test_the_retry_and_the_repair(written):
    run, llm, calls, _ = written
    first = [c for c in cands(run) if c["group"] == "ACTION3" and c["origin"] == "first round"]
    assert len(first) == 4
    ran_out = next(c for c in first if c["finished"] is False)
    assert ran_out["retried"] and ran_out["stage"] == "checked"          # the code-only retry wrote the rule
    repaired = [c for c in first if c["retried"] and c["finished"]]
    assert repaired and repaired[0]["stage"] == "checked"                 # the broken rule was repaired
    assert sum(c["duplicate"] for c in first) >= 1                        # the same code twice is one rule
    assert llm["ran_out"] == 1 and llm["candidates"] == len(cands(run))


def test_candidates_carry_what_s3_and_s4_need(written, tmp_path):
    run, *_ = written
    for c in cands(run):
        assert c["source"] == "s2_write" and isinstance(c["id"], int)
        if c["stage"] == "checked":
            assert c["binding"] is not None and "trust" in c and isinstance(c["primitives"], list)
    numbers = s3_check.run_game(run, "maze", log=lambda m: None, force=True)
    assert numbers["rules"] >= 4 and numbers["wrong"] == 0
    t = s4_carry.run_game(run, "maze", log=lambda m: None, force=True)
    assert t["t1"]["rules_in"] >= 4 and t["t1"]["wrong"] == 0


def test_a_rerun_from_the_cache_asks_nothing(written):
    run, llm, calls, _ = written
    n = len(calls)
    client = ChatClient(cache_path=os.path.join(run.dir, "s2_write", "llm_cache.jsonl"))
    client._served = "mock"

    def refuse(messages, n):
        raise AssertionError("not in the cache")
    client.mock = refuse
    again = W.run_all(run, ["maze"], client, log=lambda m: None)
    assert again["maze"]["candidates"] == llm["candidates"] and len(calls) == n
