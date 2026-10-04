"""Unit tests for the Coach's advice handling (custom_agents/coach/advice.py) and
the offline probe's scoring (tools/advice_probe.py). No LLM needed.

    uv run python -m pytest tests/ -q
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from coach.advice import (heuristic_advice, random_advice, ranking, schema,  # noqa: E402
                          validate)
from advice_probe import (hit, poisson_binomial_sf, random_expectation,  # noqa: E402
                          sign_test_two_sided)


def summary(n_obj=4, buttons=(0, 1, 2, 3), clicks=True):
    objs = [{"id": i, "colour": i, "bbox": [i, i, i, i], "size": 1, "target": i, "label": i,
             "tried": 0, "changed": 0, "new": 0} for i in range(n_obj)]
    return {"text": "x", "objects": objs, "clicks": clicks,
            "buttons": [{"action": a, "uses": 0, "changed": 0, "new": 0} for a in buttons]}


def test_schema_limits_ids_and_buttons():
    sc = schema(summary(3, (0, 2)))
    assert sc["properties"]["click_targets"]["items"]["properties"]["id"]["enum"] == [0, 1, 2]
    assert sorted(sc["properties"]["action_weights"]["required"]) == ["ACTION1", "ACTION3"]
    assert schema(summary(3, (0,), clicks=False))["properties"]["click_targets"]["maxItems"] == 0


def test_validate_drops_unknowns_and_clamps():
    adv, probs = validate({"hypothesis": "h", "click_targets": [{"id": 1, "weight": 9}, {"id": 7, "weight": 2}],
                           "action_weights": {"ACTION1": -3, "ACTION5": 2},
                           "avoid": {"objects": [2, 9], "actions": ["ACTION1"]}}, summary())
    assert adv["click_targets"] == [{"id": 1, "weight": 5}]
    assert adv["action_weights"] == {"ACTION1": 0}
    assert adv["avoid"] == {"objects": [2], "actions": ["ACTION1"]}
    assert len(probs) == 2
    assert validate("nope", summary())[0] is None


def test_ranking_orders_by_weight_and_skips_avoided_and_zero():
    adv = {"hypothesis": "", "click_targets": [{"id": 0, "weight": 2}, {"id": 1, "weight": 5},
                                               {"id": 2, "weight": 5}, {"id": 3, "weight": 0}],
           "action_weights": {"ACTION1": 3}, "avoid": {"objects": [2], "actions": []}}
    objs, aw = ranking(adv, summary())
    assert objs == [1, 0] and aw == {"ACTION1": 3}


def test_heuristic_prefers_untried_then_new_rate():
    s = summary(4)
    s["objects"][0].update(tried=10, new=5)
    s["objects"][1].update(tried=10, new=0)
    s["buttons"][1].update(uses=50, new=0)
    objs, aw = ranking(heuristic_advice(s), s)
    assert objs[:2] == [2, 3] and objs[2] == 0          # untried first, then the better tried one
    assert aw["ACTION1"] == 5 and aw["ACTION2"] == 1      # never pressed > pressed with no new screens


def test_random_advice_is_valid_and_seeded():
    s = summary()
    a1, a2 = random_advice(s, random.Random(3)), random_advice(s, random.Random(3))
    assert a1 == a2 and validate(a1, s)[1] == []


def test_hit_and_random_expectation():
    p = {"target": {"kind": "object", "id": 2}, "summary": summary(4)}
    assert hit(p, [0, 1, 2], {}, 0) == 1 and hit(p, [0, 1, 3, 2], {}, 0) == 0
    assert random_expectation(p) == 0.75
    b = {"target": {"kind": "button", "button": "ACTION2"}, "summary": summary(4)}
    assert hit(b, [], {"ACTION1": 1, "ACTION2": 4}, 0) == 1 and random_expectation(b) == 0.25
    tied = {"ACTION1": 4, "ACTION2": 4}
    assert hit(b, [], tied, 5) == hit(b, [], tied, 5)     # the same draw every time


def test_statistics():
    assert abs(poisson_binomial_sf([0.5] * 4, 4) - 1 / 16) < 1e-12
    assert poisson_binomial_sf([0.2, 0.3], 0) == 1.0
    assert abs(sign_test_two_sided(0, 5) - 2 / 32) < 1e-12
    assert sign_test_two_sided(3, 3) == 1.0
