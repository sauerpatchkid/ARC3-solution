"""tools/wm_thinkcap.py: how a recorded answer is cut at a token cap. No GPU."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

import wm_thinkcap as W  # noqa: E402


def test_an_answer_is_cut_only_inside_its_thinking_and_only_when_too_long():
    rate = 3.0
    short = "a" * 90 + "</think>```python\ncode\n```"
    assert W.cut(short, 100, rate) is None                     # 113 characters fit in 100 tokens x 3
    long = "t" * 600 + "</think>the answer and its code"
    assert W.cut(long, 100, rate) == "t" * 300                 # what a cap of 100 tokens leaves: thinking only
    unfinished = "t" * 900                                     # ran out inside its thinking
    assert W.cut(unfinished, 100, rate) == "t" * 300


def test_outcomes_read_the_v1_and_v2_grades():
    assert W.outcome_v1(None) == {"runs": False, "admitted": False, "exact": False}
    assert W.outcome_v1({"stage": "checked", "grade": {"admitted": True, "plan_eligible": False}}) == \
        {"runs": True, "admitted": True, "exact": False}
    assert W.outcome_v2({"stage": "checked", "trust": {"admitted": True, "trusted": True}}) == \
        {"runs": True, "admitted": True, "exact": True}
    assert W.outcome_v2({"stage": "static"})["runs"] is False
