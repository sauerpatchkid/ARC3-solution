"""custom_agents/kp_bridge.py: the groupmate's planner run unedited inside Goose.

The games are Toggle and, for the explorer, Walk (tests/kp_games.py). `drive`
makes the calls action.py's lines tagged `# [kp]` make, in their order, around
one of them.
"""
import ast
import hashlib
import json
import os
import random
import sys

import numpy as np
import pytest

pytest.importorskip("arc_agi")      # the reference imports it

import torch  # noqa: E402

import kp_bridge as K  # noqa: E402
import kp_games as G  # noqa: E402
from conftest import ROOT  # noqa: E402

PATTERN_RULE = "AA|pattern|None|cell|learned"
E21 = {"bridge": {"plan": True, "parse_cache": 4096}, "see": {"mask": "none"}, "explore": {"keys": "e21"}}
HASHES = {      # custom_agents/kp_ref/README.md: the files as received
    "e21_object_explorer.py": "76b262c953c603399b3fa4522c965e577e273813e9ecc9616c3410f2750d5be9",
    "e18_knowledge_planner.py": "bc48c989e7431c40e59ba2511870b1d89f57171bf5a17e7c9a48623a94605cd4",
    "e19_adaptive_planner.py": "fdf5b446434efaa4b824e646fb45033efcbab111a68987f42e8834e6fffeac83",
    "e20_learned_knowledge.py": "fb370914aed5b25f579adb37479ea3421e5c7452e97ac2accac1b8d10d0b6836",
    "knowledge_bank.py": "77a917efeb260de50c7c205c263b27a1105033c8209f7e1861781dd496dfa280",
}


@pytest.fixture(autouse=True)
def reference_as_loaded():
    """Each test leaves the reference as it was loaded: the three replacements are process-wide."""
    yield
    K.uninstall(K.load_reference()[0])


def bridge(tmp_path, **kw):
    return K.Bridge(str(tmp_path / "kp"), **kw)


def unpack(index):
    """The agent's index as the triple Goose's sampler returns."""
    return (index, None, None) if index < 5 else (5, divmod(index - 5, G.GRID), index - 5)


def drive(b, game, goose=G.LEVEL_1, steps=400, default=G.DEAD):
    """Play `game` with the hook's calls. Goose's moves come from `goose`, then `default`.
    Returns [(the move sent, did the reference take it)]."""
    goose, sent, obs, counter = list(goose), [], game.reset(), 0
    while counter < steps and obs.state.name != "WIN":
        if obs.state.name == "GAME_OVER":
            b.on_game_over()
            obs, counter = game.reset(), counter + 1
            continue
        b.step(obs.frame, obs.levels_completed, obs.available_actions, None, counter)
        move = goose.pop(0) if goose else default
        action, _, coord = b.override(*unpack(move))
        final = action if action < 5 else 5 + coord
        b.record(final)
        sent.append((final, b.on and b.took))
        obs, counter = game.play(final), counter + 1
    return sent


# ---- it plays ------------------------------------------------------------------------------------------
def test_the_reference_plans_the_levels_after_the_first_through_the_bridge(tmp_path):
    b, game = bridge(tmp_path), G.Toggle()
    sent = drive(b, game)
    assert game.level == 3 and b.n["kp_errors"] == 0                    # the game is won
    assert sent[:len(G.LEVEL_1)] == [(m, False) for m in G.LEVEL_1]     # level 1 is Goose's, move for move
    assert all(planned for _, planned in sent[len(G.LEVEL_1):])         # every move after it is the reference's
    assert b.agent.store.ranked() == [PATTERN_RULE]                     # the one rule level 1 left standing
    s = b.summary(won=True)
    assert [v["ended"] for v in s["levels"]] == ["cleared by Goose", "cleared by planning", "cleared by planning"]
    assert s["levels_by_planning"] == 2 and s["levels_by_goose"] == 1 and s["fallbacks"] == 0
    assert s["screens_that_fit"] == s["screens"] and s["clicks"]["probe"] == 18      # 8 tiles and the pattern, twice
    assert sum(s["clicks"].values()) == len(sent) - len(G.LEVEL_1) == len(s["planned_at"])
    assert s["game_overs"] == 2                 # one in Goose's level 1, one that cut level 2's plan: model kept, replanned
    first, second = (v["learned"] for v in s["levels"][:2])             # what the reference took from each level clear
    assert first["solved_board_fits"] and first["rules_consistent"] == 1 and first["negatives"] == 8
    assert (first["slots"], first["dynamic_slots"], first["dynamics"], first["cycle"]) == (9, 8, "recolour", 2)
    assert first["carried_rule"] == "" and second["carried_rule"] == "confirmed again" and second["top_rule"] == PATTERN_RULE
    assert "learned" not in s["levels"][2]      # the game ends at the last level-up: no step follows it


def test_the_bridge_makes_the_clicks_the_reference_makes_in_its_own_harness(tmp_path):
    ours = drive(bridge(tmp_path), G.Toggle())
    e18, _, e20 = K.load_reference()
    K.uninstall(e18)                            # the reference as loaded, with its explorer following the same script
    script, e0_act = list(G.LEVEL_1), e18.StochasticGooseAgent.act

    def scripted(self, obs, legal):
        self._check_level_up(obs)
        self.step_count += 1
        self.level_actions += 1
        y, x = divmod((script.pop(0) if script else G.DEAD) - 5, G.GRID)
        return legal[0], {"x": x, "y": y}, "scripted", True

    e18.StochasticGooseAgent.act = scripted
    try:
        arc = G.Arcade()
        e18.run_episode(arc, "toggle", e20.LearnedKnowledgeAgent(seed=0), None, max_steps=400, seed=0)
    finally:
        e18.StochasticGooseAgent.act = e0_act
    theirs = [(G.click(d["y"], d["x"]), reason["raw"] != "scripted") for _, d, reason in arc.env.sent]
    assert theirs == ours and arc.env.level == 3


def test_planning_switched_off_watches_and_learns_and_never_takes_a_move(tmp_path):
    b, game = bridge(tmp_path, plan=False), G.Toggle()
    sent = drive(b, game, steps=60)
    assert not any(planned for _, planned in sent) and game.level == 1          # Goose's dead clicks never solve level 2
    assert b.agent.store.ranked() == [PATTERN_RULE] and b.n["kp_errors"] == 0   # but level 1 was learned from
    assert sum(b.summary()["clicks"].values()) == 0 and b.config()["kp_plan"] is False
    assert b.summary()["levels"][1]["gave_up"] == "planning is switched off"


def test_a_level_the_planner_never_engages_on_says_what_it_lacked(tmp_path):
    b, game = bridge(tmp_path), G.Toggle()
    drive(b, game, goose=[G.DEAD] * 12 + [G.tile(dy, dx) for dy, dx in G.RING[:7]], steps=20)     # one move short of level 1's end
    b.agent.level_transitions.clear()           # as on a game whose level 1 never fitted its slots:
    b.agent.pending = None                      # no click was seen to do anything
    obs = game.play(G.tile(*G.RING[7]))
    b.step(obs.frame, obs.levels_completed, obs.available_actions, None, 20)
    assert b.override(*unpack(G.DEAD)) == unpack(G.DEAD) and game.level == 1
    assert b.level["level"] == 2 and b.level["gave_up"] == "no clickable kind learned (K1)" and b.n["fallbacks"] == 0


def test_a_game_without_clicks_never_wakes_the_reference(tmp_path):
    b, game = bridge(tmp_path), G.Toggle(clicks=False)
    sent = drive(b, game, goose=[], steps=30, default=0)
    assert sent == [(0, False)] * len(sent) and len(sent) > 20
    assert b.agent.step_count == 0 and b.cache.calls == 0 and b.summary()["dormant"]


# ---- the explorer (E21), on a game with keys -------------------------------------------------------------
def test_the_explorer_plays_the_levels_after_the_first_on_a_game_with_keys(tmp_path):
    b, game = bridge(tmp_path, settings=E21), G.Walk()
    sent = drive(b, game, goose=G.WALK_LEVEL_1, default=G.UP)
    assert game.level == 3 and b.n["kp_errors"] == 0                              # the game is won
    assert sent[:len(G.WALK_LEVEL_1)] == [(m, False) for m in G.WALK_LEVEL_1]     # level 1 is Goose's, move for move
    assert all(took for _, took in sent[len(G.WALK_LEVEL_1):])                    # every move after it is the explorer's
    assert all(move < 5 for move, _ in sent)                                      # and it sends keys, as Goose's index
    s = b.summary(won=True)
    assert s["keys"] and [v["ended"] for v in s["levels"]] == ["cleared by Goose"] + ["cleared by the explorer"] * 2
    assert s["levels_by_explorer"] == 2 and s["levels_by_planning"] == 0 and s["planned_not_sent"] == 0
    assert s["moves"] == {"explorer": len(sent) - len(G.WALK_LEVEL_1), "goose": len(G.WALK_LEVEL_1)}
    assert s["levels"][0]["moves"] == {"goose": len(G.WALK_LEVEL_1)} and set(s["levels"][1]["moves"]) == {"explorer"}
    learned = s["explorer_learned"]                                               # what E21 took from the levels it saw cleared
    assert learned["KM_controlled_kind"].startswith(f"c{G.PIECE}:") and len(learned["level_history"]) == 2
    assert [g["kind"].split(":")[0] for g in learned["KG_goal_kinds_ranked"]] == [f"c{G.GOAL}"]


def test_with_the_explorer_a_game_with_clicks_only_is_still_the_planners(tmp_path):
    plain = drive(bridge(tmp_path / "a"), G.Toggle())
    b, game = bridge(tmp_path / "b", settings=E21), G.Toggle()
    assert drive(b, game) == plain and game.level == 3                # the same moves, and the game is won
    s = b.summary(won=True)
    assert not s["keys"] and s["levels_by_planning"] == 2 and s["levels_by_explorer"] == 0 and s["explorer_learned"] is None
    assert b.agent.store.ranked() == [PATTERN_RULE]


def test_with_planning_off_the_explorer_watches_and_never_takes_a_move(tmp_path):
    b, game = bridge(tmp_path, settings=E21, plan=False), G.Walk()
    sent = drive(b, game, goose=G.WALK_LEVEL_1, default=G.UP, steps=40)
    assert not any(took for _, took in sent) and game.level == 1      # Goose's key never reaches level 2's goal
    assert b.summary()["explorer_learned"]["KT_level_ending_actions"] == {"ACTION4": 1} and b.n["kp_errors"] == 0


def test_the_explorer_books_a_click_of_gooses_by_the_piece_under_it(tmp_path):
    b, game = bridge(tmp_path, settings=E21), G.Walk(clicks=True)
    obs = game.reset()
    for counter, (move, kind) in enumerate([(G.click(9, 9), f"c{G.PIECE}"), (G.click(2, 2), "background")]):
        b.step(obs.frame, 0, obs.available_actions, None, counter)
        assert b.override(*unpack(move)) == unpack(move) and not b.took           # level 1: Goose's click stands
        b.record(move)
        y, x = divmod(move - 5, G.GRID)
        assert b.agent.last[1].startswith(f"click:{kind}") and b.agent.last[1].endswith(f"@{x},{y}")
    b.step(obs.frame, 0, obs.available_actions, None, 2)
    b.override(G.UP, None, None)
    b.record(G.DOWN)                                                  # another move was sent than the one E21 booked
    assert b.agent.last is None and b.n["planned_not_sent"] == 1 and b.n["kp_errors"] == 0


def test_without_the_explorer_a_game_with_keys_never_wakes_the_reference(tmp_path):
    b, game = bridge(tmp_path), G.Walk()
    sent = drive(b, game, goose=G.WALK_LEVEL_1, default=G.UP, steps=40)
    assert not any(took for _, took in sent) and b.agent.step_count == 0 and b.summary()["dormant"]
    assert "explorer" not in b.summary()                              # and what a run records is as before


def test_the_explorer_draws_no_random_number(tmp_path):
    before = (torch.random.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    b, game = bridge(tmp_path, settings=E21), G.Walk()
    drive(b, game, goose=G.WALK_LEVEL_1, default=G.UP)
    assert game.level == 3
    assert torch.equal(torch.random.get_rng_state(), before[0])
    assert np.array_equal(np.random.get_state()[1], before[1]) and random.getstate() == before[2]


# ---- moves in and out ----------------------------------------------------------------------------------
def test_a_click_is_converted_both_ways(tmp_path):
    b, game, booked = bridge(tmp_path), G.Toggle(), []
    obs = game.reset()
    b.step(obs.frame, 0, obs.available_actions, None, 0)
    assert b.override(*unpack(G.click(9, 7))) == (5, (9, 7), 64 * 9 + 7)       # Goose's move stands on level 1
    b.agent._record_click = lambda grid, data: booked.append(data)
    b.record(G.click(9, 7))
    assert booked == [{"x": 7, "y": 9}]                                         # booked at column 7, row 9
    b.step(obs.frame, 0, obs.available_actions, None, 1)
    b.agent.act = lambda obs, legal: (legal[0], {"x": 40, "y": 21}, "E20_learned_knowledge:plan(40,21)", True)
    assert b.override(2, None, None) == (5, (21, 40), 64 * 21 + 40)             # a planned click replaces a key press
    b.record(G.click(21, 40))
    assert len(booked) == 1 and b.n["plan"] == 1 and b.n["planned_not_sent"] == 0


def test_a_key_press_is_not_booked(tmp_path):
    b, game, booked = bridge(tmp_path), G.Toggle(), []
    obs = game.reset()
    b.step(obs.frame, 0, obs.available_actions, None, 0)
    assert b.override(3, None, None) == (3, None, None)
    b.agent._record_click = lambda grid, data: booked.append(data)
    b.record(3)
    assert booked == [] and b.agent.pending is None


def test_a_move_no_hook_saw_drops_the_pending_click(tmp_path):
    b, game = bridge(tmp_path), G.Toggle()
    obs = game.reset()
    for counter, skip in ((0, False), (1, False), (3, True)):       # Goose's counter jumps from 1 to 3: a move went by unseen
        b.step(obs.frame, 0, obs.available_actions, None, counter)
        assert (b.agent.pending is None) == (counter != 1) and b.n["skipped_steps"] == int(skip)
        b.override(*unpack(G.tile(0, 1)))
        b.record(G.tile(0, 1))
        assert b.agent.pending is not None                           # Goose's click waits for its result
    b.on_game_over()                                                 # a game over's RESET is a move too, and not a skipped one
    b.step(obs.frame, 0, obs.available_actions, None, 5)
    assert b.n["skipped_steps"] == 1 and b.agent.pending is None


# ---- it cannot hurt the run ----------------------------------------------------------------------------
def test_an_error_inside_the_reference_leaves_the_run_alive_and_is_counted_once(tmp_path):
    b, game = bridge(tmp_path), G.Toggle()

    def broken(obs, legal):
        raise RuntimeError("inside the reference")

    b.agent.act = broken
    sent = drive(b, game, steps=30)
    assert sent == [(m, False) for m in (G.LEVEL_1 + [G.DEAD] * 30)[:len(sent)]]      # Goose played on
    assert b.n["kp_errors"] == 1 and not b.on
    assert "inside the reference" in [e for e in b.events if "traceback" in e][0]["traceback"]


def test_the_bridge_draws_no_random_number(tmp_path):
    before = (torch.random.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    b, game = bridge(tmp_path), G.Toggle()
    drive(b, game)                                                   # built, and three levels cleared
    assert game.level == 3
    assert torch.equal(torch.random.get_rng_state(), before[0])
    assert np.array_equal(np.random.get_state()[1], before[1]) and random.getstate() == before[2]


def test_the_pandas_stand_in_is_gone_once_the_reference_is_loaded():
    K.load_reference()
    if "pandas" in sys.modules:
        pytest.skip("pandas is installed here: no stand-in is made")
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])           # torch looks pandas up by name; a module with no spec stops it


def test_kp_with_glassbox_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_GB", "1")
    with pytest.raises(SystemExit, match="EVAL_GB"):
        K.from_env("ft09", 0, str(tmp_path / "kp"))


# ---- settings and what a run writes --------------------------------------------------------------------
def test_from_env_reads_the_settings_and_the_plan_switch(tmp_path, monkeypatch):
    assert K.load_settings() == K.DEFAULTS                           # kp.toml as committed is the reference's behaviour
    monkeypatch.delenv("EVAL_GB", raising=False)
    monkeypatch.setenv("EVAL_KP_PLAN", "0")
    b = K.from_env("ft09", 0, str(tmp_path / "kp"))
    assert b.config() == {"kp": True, "kp_plan": False, "kp_config_hash": K.digest(K.DEFAULTS), "kp_settings": K.DEFAULTS}
    other = tmp_path / "other.toml"
    other.write_text('[see]\nmask = "agent"\n')
    assert K.digest(K.load_settings(str(other))) != K.digest(K.DEFAULTS)
    other.write_text("[see]\nwin_frame = 1\n")
    with pytest.raises(SystemExit, match="win_frame"):               # a setting no code reads is refused
        K.load_settings(str(other))


def test_the_settings_the_planners_test_ran_with_keep_their_hash(tmp_path):
    assert K.digest(K.load_settings()) == "d468480ec9387b71"         # docs/4-knowledge-planner/kp-test.md, "What is frozen"
    with_explorer = K.load_settings(os.path.join(ROOT, "experiments", "kp", "kp_e21.toml"))
    assert with_explorer == E21 and K.digest(with_explorer) != K.digest(K.DEFAULTS)
    other = tmp_path / "other.toml"
    other.write_text('[explore]\nkeys = "e22"\n')
    with pytest.raises(SystemExit, match="keys"):                    # only an explorer that is here can be asked for
        K.load_settings(str(other))


def test_the_agent_mask_is_painted_over_before_a_screen_is_parsed(tmp_path):
    b, game = bridge(tmp_path), G.Toggle()
    screen = game.reset().frame[0].astype(int)
    screen[60:63, 5:9] = 7                                           # a counter the reference's own border rule does not catch
    assert len(b.cache(screen)[0]) == 10
    b.cache.mask = np.zeros((G.GRID, G.GRID), bool)
    b.cache.mask[60:63, 5:9] = True
    assert len(b.cache(screen)[0]) == 9 and screen[60, 5] == 7      # nine objects again, and the caller's screen is untouched


def test_the_parse_memory_returns_what_the_reference_would_parse(tmp_path):
    b, game = bridge(tmp_path), G.Toggle()
    screen = game.reset().frame[0].astype(int)
    first, again = b.cache(screen), b.cache(screen)
    real = K._REPLACED["parse"](screen)
    assert b.cache.calls == 2 and b.cache.parsed == 1 and first[0] is not again[0]
    assert [(o.pos, o.sig, o.click) for o in again[0]] == [(o.pos, o.sig, o.click) for o in real[0]] and again[1] == real[1]


def test_dump_writes_the_totals_and_the_events(tmp_path):
    b, game = bridge(tmp_path), G.Toggle()
    drive(b, game)
    (tmp_path / "run_end.json").write_text(json.dumps({"termination": "win"}))
    b.dump(str(tmp_path / "kp" / "kp_stats.json"))
    stats = json.load(open(tmp_path / "kp" / "kp_stats.json"))
    assert stats["levels_by_planning"] == 2 and stats["kp_errors"] == 0 and stats["levels"][2]["ended"] == "cleared by planning"
    events = [json.loads(line) for line in open(tmp_path / "kp" / "events.jsonl")]
    assert {"reference", "learned", "level ended"} <= {e["event"] for e in events}
    assert [e["entry"][0] for e in events if e["event"] == "reference" and e["level"] == 2][0] == "probe-set"
    assert json.load(open(tmp_path / "kp" / "store.json"))["K3_goal_hypotheses_ranked"][0]["key"] == PATTERN_RULE


# ---- the repo's promises ---------------------------------------------------------------------------------
def test_the_reference_files_are_the_ones_received():
    for name, sha in HASHES.items():
        with open(os.path.join(ROOT, "custom_agents", "kp_ref", name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == sha, f"{name} was edited"


def test_the_hook_in_goose_is_tagged_and_comes_out_with_one_sed():
    lines = open(os.path.join(ROOT, "custom_agents", "action.py")).read().split("\n")
    tagged = [line for line in lines if "# [kp]" in line]
    assert 15 <= len(tagged) <= 22
    without = "\n".join(line for line in lines if "# [kp]" not in line)
    ast.parse(without)                                               # sed -i '/\[kp\]/d' leaves a file that still parses
    assert "kp" not in without.replace("skip", "").lower().replace("backprop", "")
