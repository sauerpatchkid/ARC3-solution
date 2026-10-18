"""S4 CARRY version 1 (custom_agents/gb/s4_carry.py): the role tie-break and the online
wait. The re-binding itself is CP1's (tests/test_gb_regression.py checks it on dc22)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import s4_carry as S  # noqa: E402

MOVER_RULE = '''
RULE = "the agent moves along the track"
PARAMS = {"agent": Colour(), "track": Colour(), "step": Int(1, 8)}
def applies(board, act, api, p):
    return act.action == 1
def predict(board, act, api, p):
    return board.copy(), board == p["agent"]
'''
# m0r0, G1: colour 10 moves on both levels; 5 is static on both; 6 is static on level 2.
ROLES_L1 = {"mover": 10, "movers": [10], "static": [5, 11, 12], "click_changers": [], "click_noops": [0, 5, 11, 12],
            "counters": []}
ROLES_L2 = {"mover": 10, "movers": [10], "static": [5, 6, 8, 15], "click_changers": [],
            "click_noops": [0, 5, 6, 8, 15], "counters": []}


def test_colour_slots_are_read_from_params_without_running_the_rule():
    assert S.colour_slots(MOVER_RULE) == ["agent", "track"]
    assert S.colour_slots("def applies(b, a, api):\n    return False\n") == []


def test_role_sets_and_scores():
    assert S.role_set(ROLES_L1, 10) == {"mover"}
    assert S.role_set(ROLES_L2, 6) == {"static", "click_noops"}
    before = {"agent": 10, "track": 5, "step": 5}
    keeps = {"agent": 10, "track": 5, "step": 4}
    moves_agent = {"agent": 6, "track": 5, "step": 5}
    assert S.role_score(["agent", "track"], before, keeps, ROLES_L1, ROLES_L2) == 2
    assert S.role_score(["agent", "track"], before, moves_agent, ROLES_L1, ROLES_L2) == 1


def test_by_roles_keeps_the_search_choice_when_roles_do_not_prefer_another():
    c = {"code": MOVER_RULE, "binding": {"agent": 10, "track": 5, "step": 5}}
    best = {"binding": {"agent": 10, "track": 5, "step": 4}, "res": None}
    found = {"exact_list": [{"binding": {"agent": 10, "track": 5, "step": 4}, "applies": 22},
                            {"binding": {"agent": 6, "track": 5, "step": 5}, "applies": 22}]}
    got, note = S.by_roles(c, found, best, None, (ROLES_L1, ROLES_L2))
    assert got is best and note["chosen"] == "search" and note["score"] == 2


def test_the_online_wait():
    s = dict(S.DEFAULTS)
    assert not S.ready(999, 9, s)
    assert S.ready(999, 10, s) and S.ready(1000, 0, s)


def test_roles_are_off_by_default_so_s4_is_cp1s():
    class Run:
        config = {"carry": {}}
    assert S.settings_of(Run()) == (dict(S.DEFAULTS), {})
    Run.config = {"carry": {"roles_in_bind": True}}
    assert S.settings_of(Run())[1] == {"roles_in_bind": True}


def test_by_roles_never_prefers_a_binding_that_applies_to_fewer_cases():
    """ft09 (pool, 18 Oct): target 12 applies to 6 fit cases, target 9 to 2. 9 keeps its
    role (a click changer on both levels), 12 does not quite (also a counter on level 2),
    but the binding that applies to more cases stays."""
    rule = MOVER_RULE.replace('"agent": Colour(), "track": Colour(), "step": Int(1, 8)', '"target": Colour()')
    c = {"code": rule, "binding": {"target": 9}}
    best = {"binding": {"target": 12}, "res": None}
    found = {"exact_list": [{"binding": {"target": 12}, "applies": 6}, {"binding": {"target": 9}, "applies": 2}]}
    r1 = {"click_changers": [8, 9], "counters": [11, 12]}
    r2 = {"click_changers": [9, 12], "counters": [11, 12]}
    got, note = S.by_roles(c, found, best, None, (r1, r2))
    assert got is best and note is None
