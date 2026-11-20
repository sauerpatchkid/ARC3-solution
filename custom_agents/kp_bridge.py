"""kp_bridge.py - the groupmate's planner (E20), run unedited inside this project's Goose.

The problem it fixes: the reference agent (custom_agents/kp_ref/) expects its
own harness and its own small Goose. This runs it inside the project's Goose,
whichever upgrades are on: it hands the reference each screen, takes its
planned clicks, and tells it which move Goose made. Goose plays level 1 and is
the fallback; the reference watches, learns at every level clear, and plans the
levels after the first. Plan: docs/4-knowledge-planner/kp-plan.md.

    EVAL_KP=1                 on. Unset (the default), not one line of this file runs
    EVAL_KP_PLAN=0            watch and learn only: the reference never takes a move
    EVAL_KP_CONFIG=<path>     another settings file (default experiments/kp/kp.toml)
    [explore] keys = "e21"    in that file (experiments/kp/kp_e21.toml has it; kp.toml does not): the
                              groupmate's explorer E21 plays the levels after the first on games with keys

The reference's files are never edited. install() replaces three functions on
the loaded module:

    StochasticGooseAgent.act                its explorer answers "this move is Goose's"
    StochasticGooseAgent._new_level_model   does nothing. It built a CNN nobody uses and reseeded
                                            torch's global generator, which Goose itself draws from
    parse_objects                           the same function behind a memory of the screens already
                                            parsed (the reference parses one screen 3 to 5 times a move)

With those in place the reference draws no random number, so given the same
screens it makes the same decisions, and Goose's own random streams run as they
would without it until its first planned click.

One step, in the order action.py calls the hook (its lines tagged `# [kp]`):

    step()           the last move's result: the whole frame stack, because a level-up's
                     solved board is the frame before the last
    override()       after Goose has sampled and every other override has run: ask the reference.
                     A probe, a verify press or a plan step takes the move; otherwise Goose's stands
    record()         the move actually sent. Goose's own clicks are booked the way the reference
                     books its explorer's
    on_game_over()   the reference drops its plan and keeps its model

Until a game first offers a click (ACTION6) the reference is not called at
all: on a game without clicks the agent is Goose, move for move.

The explorer (added 20 Nov; docs/4-knowledge-planner/kp-explorer.md). The
problem it fixes: E20 watches clicks only, so on a game played with keys the
planner does nothing, and Goose alone rarely clears a second level there.
With `[explore] keys = "e21"` the agent is E21 (kp_ref/e21_object_explorer.py),
a subclass of E20 that hands a game with clicks only to E20 unchanged. On a
game with keys E21 tries every move it has not tried on a screen, walks back
to the nearest screen that still has one, and carries what a cleared level
taught to the next. It chooses a move in one place, its `_pick`, and the
bridge stands there (pick): level 1 is Goose's move, a later level is E21's
own until it has nothing left to try. E21 books the move either way, so it
learns from Goose's level 1 as it would from its own. The reference is then
called on every game, with clicks or without.

A run writes <run folder>/kp/: store.json (the reference's own knowledge file,
rewritten at each level clear), events.jsonl (the reference's two logs with
the step each line came at, and the bridge's own lines) and kp_stats.json
(totals, and one record per level).

Any error in the reference or in this file is caught, written to events.jsonl
with its traceback, counted (`kp_errors`), and switches the planner off for the
rest of the run. Goose plays on; the run is named in the write-ups.
"""
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import tomllib
import traceback
from collections import Counter, OrderedDict

import numpy as np

from eval_common import env_flag
from gridtools import GRID

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "kp_ref")
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "experiments", "kp", "kp.toml")
DEFAULTS = {"bridge": {"plan": True, "parse_cache": 4096}, "see": {"mask": "none"}, "explore": {"keys": "none"}}
MASKS = ("none", "agent")
EXPLORERS = ("none", "e21")
LATER = ("explore",)        # sections added after the planner's test of 19 Nov (see digest)
CLICK = "ACTION6"
MOVES = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", CLICK)      # what Goose can send, in its own order
GOOSE = object()            # what the stand-in explorer returns: "this move is Goose's"


# ---- the reference -----------------------------------------------------------------------------------
def load(*names):
    """The reference's modules of these names, imported as they are. They import
    pandas at the top and use it only in their own main(): where pandas is not
    installed, an empty stand-in takes its name while they load, so no
    dependency is added. The stand-in is taken away again at once: torch looks
    pandas up by name the first time an optimizer is built, and a module left
    in sys.modules with no spec stops it ("pandas.__spec__ is None")."""
    if REF not in sys.path:
        sys.path.insert(0, REF)
    import torch  # noqa: F401  (loaded before the stand-in exists)
    stand_in = "pandas" not in sys.modules and importlib.util.find_spec("pandas") is None
    if stand_in:
        sys.modules["pandas"] = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("pandas", None))
    try:
        return tuple(importlib.import_module(name) for name in names)
    finally:
        if stand_in:
            del sys.modules["pandas"]


def load_reference():
    """E18, E19 and E20: the planner."""
    return load("e18_knowledge_planner", "e19_adaptive_planner", "e20_learned_knowledge")


def load_explorer():
    """E21: the explorer for games with keys (module doc)."""
    return load("e21_object_explorer")[0]


def goose_moves(self, obs, legal):
    """In place of E0's act: E0's own bookkeeping, and no move of its own."""
    self._check_level_up(obs)
    self.step_count += 1
    self.level_actions += 1
    return GOOSE, {}, "goose", True


def no_model(self):
    """In place of E0's _new_level_model (module doc)."""


class ParseCache:
    """The reference's parse_objects behind a memory of the last `size` screens.
    Speed only: the objects of a screen are the ones the reference would build,
    and it never changes an object after building it. With `[see] mask =
    "agent"` the cells under Goose's decoration mask are painted with the
    screen's commonest colour first, so a bar or a counter is not an object."""

    def __init__(self, parse, size):
        self.parse, self.size, self.seen, self.mask = parse, size, OrderedDict(), None
        self.calls = self.parsed = 0
        self.seconds = 0.0

    def __call__(self, grid, big_frac=0.08):
        self.calls += 1
        if self.mask is not None and self.mask.any():
            grid = np.array(grid)
            grid[self.mask] = np.bincount(grid.ravel()).argmax()
        key = (grid.shape, big_frac, np.asarray(grid, dtype=np.uint8).tobytes())     # colours are 0 to 15
        hit = self.seen.get(key)
        if hit is None:
            t0 = time.perf_counter()
            hit = self.seen[key] = self.parse(grid, big_frac)
            self.seconds += time.perf_counter() - t0
            self.parsed += 1
            if len(self.seen) > self.size:
                self.seen.popitem(last=False)
        else:
            self.seen.move_to_end(key)
        objs, background = hit
        return list(objs), set(background)


_REPLACED = {}              # what install() took out, so uninstall() can put it back


def install(e18, cache_size):
    """The three replacements (module doc). Returns the parse memory."""
    if not _REPLACED:
        _REPLACED.update(act=e18.StochasticGooseAgent.act, model=e18.StochasticGooseAgent._new_level_model,
                         parse=e18.parse_objects)
    cache = ParseCache(_REPLACED["parse"], cache_size)
    e18.StochasticGooseAgent.act = goose_moves
    e18.StochasticGooseAgent._new_level_model = no_model
    e18.parse_objects = cache
    return cache


def uninstall(e18):
    """The reference as it was loaded (the offline tools and the tests use it both ways in one process)."""
    if _REPLACED:
        e18.StochasticGooseAgent.act = _REPLACED["act"]
        e18.StochasticGooseAgent._new_level_model = _REPLACED["model"]
        e18.parse_objects = _REPLACED["parse"]
        _REPLACED.clear()


def fits_slots(e18, agent, grid):
    """Is every object of this screen on one of the level's slots, and every slot taken? What the
    reference's LevelLayout.encode asks, without its side effect (it numbers new signatures)."""
    return {o.pos for o in e18.parse_objects(grid)[0]} == set(agent.layout.slots)


def why_fell_back(e18, agent, grid):
    """Why the reference handed the level to its explorer. It does not say; this
    reads its state in the order its own _plan_act gives up."""
    if not agent.k1_shapes:
        return "no clickable kind learned (K1)"
    if agent.k2_class is None:
        return "no click effect learned (K2)"
    if not agent.k3:
        return "no goal rule confirmed (K3)"
    if agent.effects is None:
        return "planning is switched off"
    if not fits_slots(e18, agent, grid):
        return "the screen left the level's slots"
    if agent.mispredictions > agent.max_mispredictions:
        return f"{agent.mispredictions} wrong predictions"
    if agent.plan_failures > agent.max_plan_failures:
        return "no plan to any rule left"
    known = {k: v for k, v in agent.effects.compile(agent.layout.slots).items() if k not in agent.effects.noop}
    if not known:
        return "no click with a known effect"
    if agent.goal is None:
        return "no rule left that fits this level"
    return "no plan, and no new board within reach"


# ---- what the reference reads off a step ---------------------------------------------------------------
class Legal:
    """An available action as the reference reads one: a name, and whether it takes a position."""

    def __init__(self, name):
        self.name = name

    def is_complex(self):
        return self.name == CLICK


class Obs:
    """An observation as the reference reads one: the frame stack and the levels completed."""

    def __init__(self, frame, levels_completed):
        self.frame, self.levels_completed = frame, levels_completed


def plain(entry):
    """One line of the reference's logs as JSON can hold it (its positions are numpy floats)."""
    return [x if isinstance(x, (int, str)) or x is None else str(x) for x in entry]


# ---- the bridge ----------------------------------------------------------------------------------------
class Bridge:
    def __init__(self, out_dir, settings=None, plan=True, game_id="", seed=None):
        self.out_dir, self.plan, self.game_id, self.seed = out_dir, plan, game_id, seed
        self.settings = settings or load_settings()
        self.e18, _, e20 = load_reference()
        self.cache = install(self.e18, self.settings["bridge"]["parse_cache"])
        self.explorer = self.settings["explore"]["keys"] == "e21"
        self.e21 = load_explorer() if self.explorer else None       # E21 is E20 on a game with clicks only
        self.agent = (self.e21.ObjectExplorerAgent if self.explorer else e20.LearnedKnowledgeAgent)(
            seed=0, knowledge_path=os.path.join(out_dir, "store.json"))
        self.agent.reset()
        if self.explorer:
            self.agent._pick = self.pick        # E21's one place where a move is chosen
        if not plan:                            # it watches and learns, and never proposes
            self.agent._plan_act = lambda grid, legal: None
        self.on, self.clicks = True, self.explorer      # off after an error; dormant until the game first offers a
        self.sampled, self.chose = None, False          # click (with the explorer, never). Goose's sampled move; did E21 choose
        self.frames, self.grid, self.score, self.legal, self.counter = None, None, 0, [], 0
        self.took, self.planned = False, None   # did the reference take this move, and which click
        self.sent, self.game_over = None, False     # Goose's counter at the last move sent; a game over since
        self.level, self.levels, self.events, self.planned_at = None, [], [], []
        self.n, self.seen_log, self.seen_knowledge, self.act_seconds = Counter(), 0, 0, 0.0

    # ---- the four hooks --------------------------------------------------------------------------------
    def step(self, frames, score, available_actions, mask, counter):
        """The last move's result, kept for override(). `counter` is Goose's action counter."""
        if not self.on:
            return
        try:
            self.frames, self.score, self.counter = frames, int(score), counter
            self.legal = [Legal(a.name) for a in available_actions
                          if a.name != "RESET" and (a.name in MOVES or not self.explorer)]     # E21 tries what it is offered
            self.clicks = self.clicks or any(a.is_complex() for a in self.legal)
            if self.settings["see"]["mask"] == "agent":
                self.cache.mask = mask
            if self.sent is not None and counter != self.sent + 1 and not self.game_over:
                self.agent.pending = None       # a move went by that no hook saw: this screen is not the pending click's result
                if self.explorer:
                    self.agent.last = None      # nor the result of the move E21 booked
                self.n["skipped_steps"] += 1
                self.event("a move went by unseen: the pending click was dropped")
            self.game_over = False
        except Exception:
            self.fail()

    def override(self, action_idx, coords, coord_idx):
        """Goose's sampled move in, the move to send out: (5, (y, x), 64*y + x) for a planned click."""
        if not (self.on and self.clicks):
            return action_idx, coords, coord_idx
        try:
            obs = Obs(self.frames, self.score)
            self.grid = self.e18.get_grid(obs)
            cleared = self.level_change()
            self.sampled, self.chose = (action_idx, coords, coord_idx), False
            t0 = time.perf_counter()
            action, data, raw, _ = self.agent.act(obs, self.legal)
            self.act_seconds += time.perf_counter() - t0
            if self.keys():                     # a game with keys under the explorer: E21's own path
                return self.explorer_move(action, data)
            self.took = action is not GOOSE
            self.after_act()
            if cleared:                         # the reference has now learned from the level it saw cleared
                self.levels[-1]["learned"] = self.learned(*cleared)
                self.event("learned from the level", cleared=self.levels[-1]["level"], record=self.levels[-1]["learned"])
            if not self.took:
                return action_idx, coords, coord_idx
            x, y = int(data["x"]), int(data["y"])
            kind = raw.split(":", 1)[1].split("(")[0]       # the reference tags its clicks probe, verify or plan
            self.n[kind] += 1
            self.level["clicks"][kind] = self.level["clicks"].get(kind, 0) + 1
            self.planned = 5 + GRID * y + x
            self.planned_at.append(self.counter)
            return 5, (y, x), GRID * y + x
        except Exception:
            self.fail()
            return action_idx, coords, coord_idx

    def record(self, sent):
        """The move actually sent: 0-4 a key, 5 + 64*y + x a click."""
        if not (self.on and self.clicks):
            return
        try:
            self.sent = self.counter
            if self.keys():
                if sent != self.planned:        # E21 booked a move that was not the one sent: it learns nothing from it
                    self.agent.last = None
                    self.n["planned_not_sent"] += 1
                    self.event("the move E21 booked was not the move sent")
            elif self.took:
                if sent != self.planned:        # this hook is the last override; if that ever changes, learn nothing from it
                    self.agent.pending, self.agent.plan = None, []
                    self.n["planned_not_sent"] += 1
                    self.event("the planned click was not the move sent")
            elif sent >= 5:                     # Goose's own click, booked as the reference books its explorer's
                y, x = divmod(int(sent) - 5, GRID)
                self.agent._record_click(self.grid, {"x": x, "y": y})
        except Exception:
            self.fail()

    def on_game_over(self):
        if not (self.on and self.clicks):
            return
        try:
            self.agent.note("GAME_OVER")
            self.game_over = True
            self.n["game_overs"] += 1
        except Exception:
            self.fail()

    # ---- the explorer (E21), on a game with keys -------------------------------------------------------
    def keys(self):
        """Is this a game with keys under the explorer? E21 settles it on its first call."""
        return self.explorer and self.agent.click_only is False

    def pick(self, node, parts, grid, legal):
        """In place of E21's _pick, where it chooses a move. Level 1 is Goose's, and so is a step on
        which the explorer has nothing left to try; a later level is the explorer's. What this
        returns E21 books as the move made, in its own words: a key's name, or
        click:<the clicked piece's kind>@x,y."""
        if self.plan and self.agent.level >= 1:
            akey = type(self.agent)._pick(self.agent, node, parts, grid, legal)
            if akey is not None:
                self.chose = True
                return akey
        action_idx, coords, _ = self.sampled
        if action_idx < 5:
            return MOVES[action_idx]
        y, x = coords
        return f"click:{self.kind_at(grid, x, y)}@{x},{y}"

    def kind_at(self, grid, x, y):
        """E21's name for the piece under a click of Goose's (it books clicks by the kind of piece)."""
        colour = int(grid[y, x])
        if colour not in self.agent.bg:
            for ys, xs in self.e18._components(grid == colour):
                if ((ys == y) & (xs == x)).any():
                    return self.e21.Part(ys, xs, colour).kind
        return "background"

    def explorer_move(self, action, data):
        """The move to send on a game with keys: the explorer's own, or Goose's as it was sampled."""
        self.took = self.chose
        who = "explorer" if self.took else "goose"
        self.level["screens"] += 1
        self.level["moves"][who] = self.level["moves"].get(who, 0) + 1
        self.n[who + "_moves"] += 1
        if not self.took:
            action_idx, _, coord_idx = self.sampled
            self.planned = action_idx if action_idx < 5 else 5 + coord_idx
            return self.sampled
        if action.is_complex():
            x, y = int(data["x"]), int(data["y"])
            self.planned = 5 + GRID * y + x
            return 5, (y, x), GRID * y + x
        self.planned = MOVES.index(action.name)
        return self.planned, None, None

    def ended(self):
        """Who had the move that cleared the level."""
        if self.keys():
            return "cleared by the explorer" if self.took else "cleared by Goose"
        return "cleared by planning" if self.agent.mode == "plan" else "cleared by Goose"

    # ---- bookkeeping -----------------------------------------------------------------------------------
    def level_change(self):
        """One record per level. Called before the reference acts, so at a level-up its mode is still
        the old level's: a level is cleared by planning if it ended with the planner in charge.
        At a level-up it returns what learned() needs from before the reference learns."""
        if self.level is not None and self.level["level"] == self.score + 1:
            return None
        before = None
        if self.level is not None:
            self.close_level(self.ended())
            if self.agent.layout is not None:
                before = (self.agent.layout, self.agent.store.ranked()[:1], len(self.agent.knowledge_log))
        self.level = dict(level=self.score + 1, entered_at=self.counter, screens=0, screens_that_fit=0,
                          clicks={}, gave_up="", gave_up_at=None, ended="", left_at=None)
        if self.explorer:
            self.level["moves"] = {}            # on a game with keys: how many moves were the explorer's, how many Goose's
        return before

    def learned(self, layout, top_before, seen):
        """What the reference took from the level it has just seen cleared: was the solved board on the
        level's slots, how many rules it left standing, and what became of the rule it came in with."""
        agent, level = self.agent, self.agent.level         # it now counts the cleared level
        stack = np.array(self.frames)
        solved = len(stack) >= 2 and ({o.pos for o in self.e18.parse_objects(stack[-2].astype(int))[0]}
                                      == set(layout.slots))
        k3 = {entry[1]: entry for entry in agent.knowledge_log[seen:]}.get("K3")
        carried = ""
        if top_before:
            rec = agent.store.stats[top_before[0]]
            carried = "refuted" if rec["refuted"] else ("confirmed again" if level in rec["levels"] else "no evidence")
        ranked, k2 = agent.store.ranked(), agent.store.k2.get(f"level {level}", {})
        return dict(slots=len(layout.slots), dynamic_slots=len(layout.dynamic_set), clicking_kinds=len(agent.k1_shapes),
                    dynamics=k2.get("family", ""), cycle=k2.get("cycle_length"), solved_board_fits=bool(solved),
                    negatives=int(str(k3[3]).split()[0]) if k3 else 0,
                    rules_consistent=int(str(k3[2]).split()[0]) if k3 else 0, rules_ranked=len(ranked),
                    top_rule=ranked[0] if ranked else "",
                    top_rule_says=agent.store.hyps[ranked[0]].describe() if ranked else "", carried_rule=carried)

    def close_level(self, ended):
        self.level.update(ended=ended, left_at=self.counter)
        self.levels.append(self.level)
        self.event("level ended", record=self.level)

    def after_act(self):
        """Count the screen, and copy what the reference just logged, with the step it came at."""
        self.level["screens"] += 1
        self.level["screens_that_fit"] += fits_slots(self.e18, self.agent, self.grid)
        for entry in self.agent.log[self.seen_log:]:
            self.event("reference", entry=plain(entry))
            if entry[0] == "fallback-to-goose":
                self.n["fallbacks"] += 1
                self.level.update(gave_up=self.why(), gave_up_at=self.counter)
            self.n["wrong_predictions"] += entry[0] == "mispredict"
        if self.level["level"] > 1 and self.level["screens"] == 1 and self.agent.mode == "goose" \
                and not self.level["gave_up"]:
            self.level.update(gave_up=self.why(), gave_up_at=self.counter)     # it never engaged on this level
        for entry in self.agent.knowledge_log[self.seen_knowledge:]:
            self.event("learned", entry=plain(entry))
        self.seen_log, self.seen_knowledge = len(self.agent.log), len(self.agent.knowledge_log)

    def why(self):
        """Why the level is Goose's: what the reference lacks or ran into, or that planning is switched off."""
        reason = why_fell_back(self.e18, self.agent, self.grid)
        return reason if self.plan or "(K" in reason else "planning is switched off"

    def event(self, what, **more):
        self.events.append(dict(step=self.counter, level=self.score + 1, event=what, **more))

    def fail(self):
        self.on = False
        self.n["kp_errors"] += 1
        self.event("error: the planner is off for the rest of the run", traceback=traceback.format_exc())

    # ---- what a run records ----------------------------------------------------------------------------
    def config(self):
        """Fields for run_config.json."""
        return {"kp": True, "kp_plan": self.plan, "kp_config_hash": digest(self.settings),
                "kp_settings": self.settings}

    def summary(self, won=False):
        levels = self.levels + ([dict(self.level, ended=self.ended() if won else "not cleared",
                                      left_at=self.counter)] if self.level is not None else [])
        explorer = dict(explorer="e21", keys=self.keys(),
                        levels_by_explorer=sum(v["ended"] == "cleared by the explorer" for v in levels),
                        moves={k: self.n[k + "_moves"] for k in ("explorer", "goose")},
                        explorer_learned=self.agent.xk.to_json() if self.keys() else None) if self.explorer else {}
        return dict(explorer, game=self.game_id, plan=self.plan, config_hash=digest(self.settings),
                    kp_errors=self.n["kp_errors"], dormant=not self.clicks,
                    levels_by_planning=sum(v["ended"] == "cleared by planning" for v in levels),
                    levels_by_goose=sum(v["ended"] == "cleared by Goose" for v in levels),
                    clicks={k: self.n[k] for k in ("probe", "verify", "plan")}, planned_at=self.planned_at,
                    wrong_predictions=self.n["wrong_predictions"], fallbacks=self.n["fallbacks"],
                    screens=sum(v["screens"] for v in levels),
                    screens_that_fit=sum(v["screens_that_fit"] for v in levels),
                    game_overs=self.n["game_overs"], skipped_steps=self.n["skipped_steps"],
                    planned_not_sent=self.n["planned_not_sent"],
                    parse=dict(calls=self.cache.calls, parsed=self.cache.parsed, seconds=round(self.cache.seconds, 3)),
                    reference_seconds=round(self.act_seconds, 3), levels=levels,
                    store_history=self.agent.store.history)

    def dump(self, path, won=None):
        """At exit: kp_stats.json at `path`, events.jsonl beside it. The last level of a won game
        ends with no further step, so run_end.json (written by the game loop) says whether the run
        was won; the offline tools, which have no such file, pass `won`."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            end = os.path.join(os.path.dirname(os.path.dirname(path)), "run_end.json")
            if won is None:
                won = os.path.exists(end) and json.load(open(end)).get("termination") == "win"
            with open(path, "w") as f:
                json.dump(self.summary(won), f, indent=1, default=str)
            with open(os.path.join(os.path.dirname(path), "events.jsonl"), "w") as f:
                for e in self.events:
                    f.write(json.dumps(e, default=str) + "\n")
        except (OSError, ValueError):
            pass


# ---- settings ------------------------------------------------------------------------------------------
def load_settings(path=DEFAULT_CONFIG):
    """kp.toml over DEFAULTS. A setting this file does not read is refused, so every setting in the
    file is read by code."""
    with open(path, "rb") as f:
        given = tomllib.load(f)
    settings = {section: dict(values) for section, values in DEFAULTS.items()}
    for section, values in given.items():
        for key, value in values.items():
            if key not in settings.get(section, {}):
                raise SystemExit(f"{path}: [{section}] {key} is not a setting kp_bridge.py reads")
            settings[section][key] = value
    if settings["see"]["mask"] not in MASKS:
        raise SystemExit(f"{path}: [see] mask must be one of {', '.join(MASKS)}")
    if settings["explore"]["keys"] not in EXPLORERS:
        raise SystemExit(f"{path}: [explore] keys must be one of {', '.join(EXPLORERS)}")
    return settings


def digest(settings):
    """A short, stable hash of the settings, for run_config.json. A section added after the planner's
    test (LATER) is hashed only where it departs from its default, so the settings that test ran
    with keep the hash it recorded (d468480ec9387b71)."""
    settings = {k: v for k, v in settings.items() if k not in LATER or v != DEFAULTS[k]}
    text = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def from_env(game_id, seed, out_dir):
    """The bridge for one run. Every run starts with an empty store."""
    if env_flag("EVAL_GB", False):
        raise SystemExit("EVAL_KP=1 with EVAL_GB=1 is refused: both steer Goose's clicks, and nothing wires them together")
    settings = load_settings(os.getenv("EVAL_KP_CONFIG", "").strip() or DEFAULT_CONFIG)
    plan = settings["bridge"]["plan"] and env_flag("EVAL_KP_PLAN", True)
    os.makedirs(out_dir, exist_ok=True)
    return Bridge(out_dir, settings, plan, game_id, seed)
