"""return_map.py — Option 1: give Goose a map and a way back.

WHAT IT DOES
Goose (with Plan B's novelty label) wants new screens but only looks one move
ahead. This module adds a map of the current level: which move led from which
screen to which. It is used two ways:

  1. STALL. When Goose goes `stall` decisions without reaching a never-seen
     screen, the map finds the nearest known screen that still has something
     untried (a button never pressed there, or a clickable screen clicked fewer
     than `click_tries` times), walks there by the shortest known route, and
     tries an untried button on arrival (or lets Goose pick a click).
  2. GAME OVER. The level restarts from its first screen. Instead of finding
     its way back by chance, Goose walks the shortest known route to the most
     recently discovered screen that still has something untried, then
     explores from there. The walk may use at most half of a typical attempt
     (the median number of moves between game overs, last 20 attempts), so
     half the budget is always left for exploring. The first smoke test showed
     why: on ft09, which ends an attempt every ~36 moves, an unlimited walk
     back averaged 31 moves and left almost nothing to explore with.

Every route step is checked: if the game lands somewhere other than where the
map said, or the planned move is not available, the route is dropped and Goose
plays normally. Moves that were followed by a game over are never used in a
route. With `EVAL_RETURN_MAP` unset the agent never constructs this class.

WHY (docs/plans/option-1-return-map.md has the full version)
  - ARC-AGI-3 preview: 2nd place (Blind Squirrel) built a state graph; 3rd
    place (arXiv 2512.24156) keeps a graph and heads for the shortest path to
    untested state-action pairs, solving a median 30/52 levels with no learning.
  - Go-Explore (Ecoffet et al., Nature 2021): agents fail hard exploration
    mainly by forgetting how to return to promising states; remember, return,
    then explore.
  - Our corpora: Goose hits game over and restarts a level every ~20-200
    actions on most games, so returning after a game over is the common case.
  - Long walks back can drift: on dc22 and g50t the game stopped matching the
    map ~12 steps into a 70-80 step walk (state the screen does not show). The
    step check catches it and Goose carries on normally.

SCREEN IDENTITY
Keys use a STICKY copy of Plan B's decoration mask: every cell the online
detector has ever flagged stays masked. The live mask flips back and forth on
the progress-bar games (tu93, vc33, lf52: 44-87 flips per 100k actions), which
would scramble every key; the sticky union settles within ~3k actions and only
ever covers the progress-bar row (row 0 or 63) on the games measured. When the
sticky mask grows, keys are no longer comparable and the map starts over.

HOW TO REMOVE IT
Leave EVAL_RETURN_MAP unset (the default) and nothing here runs. To delete it,
remove this file and tests/test_return_map.py, and delete the lines tagged
`[return-map]` in custom_agents/action.py (`grep -n return-map
custom_agents/action.py`). To archive it instead, move this file to legacy/
with its results doc.

Pure Python + numpy + xxhash; no torch.
"""
import json
import os
from collections import deque

import numpy as np
import xxhash

GRID = 64
DEAD = -1                  # edge marker: this move was followed by a game over

DEFAULT_STALL = 200        # EVAL_MAP_STALL: decisions without a new screen before routing
DEFAULT_CLICK_TRIES = 20   # EVAL_MAP_CLICK_TRIES: a clickable screen is "untried" below this
DEFAULT_MAX_ROUTE = 100    # EVAL_MAP_MAX_ROUTE: longest route the map will walk
DEFAULT_MAX_NODES = 300_000


class ReturnMap:
    def __init__(self, stall=DEFAULT_STALL, click_tries=DEFAULT_CLICK_TRIES,
                 max_route=DEFAULT_MAX_ROUTE, return_after_game_over=True,
                 max_nodes=DEFAULT_MAX_NODES, rng=None):
        self.stall = int(stall)
        self.click_tries = int(click_tries)
        self.max_route = int(max_route)
        self.return_after_game_over = bool(return_after_game_over)
        self.max_nodes = int(max_nodes)
        self.rng = rng if rng is not None else np.random
        self.sticky = np.zeros((GRID, GRID), dtype=bool)   # game-level, survives levels
        self._mask_seen = None
        self.attempts = deque(maxlen=20)   # moves per attempt (between game overs), game-level
        self.attempt_steps = 0
        self.step = 0
        self.stats = {k: 0 for k in (
            "routes_started", "routes_completed", "routes_aborted",
            "return_routes", "stall_routes", "route_steps", "map_actions",
            "frontier_button_tries", "no_frontier", "game_overs",
            "mask_resets", "levels", "route_len_total",
            "aborted_mismatch", "aborted_unavailable", "aborted_game_over",
            "aborted_no_prev")}
        self.clear(new_level=False)

    # ---- construction from the eval_common-style environment flags ---------
    @classmethod
    def from_env(cls):
        return cls(stall=int(os.getenv("EVAL_MAP_STALL", DEFAULT_STALL)),
                   click_tries=int(os.getenv("EVAL_MAP_CLICK_TRIES", DEFAULT_CLICK_TRIES)),
                   max_route=int(os.getenv("EVAL_MAP_MAX_ROUTE", DEFAULT_MAX_ROUTE)),
                   return_after_game_over=os.getenv("EVAL_MAP_RETURN", "1").strip().lower()
                   in ("1", "true", "yes", "on"))

    def config(self):
        """Fields for run_config.json."""
        return {"return_map": True, "map_stall": self.stall,
                "map_click_tries": self.click_tries, "map_max_route": self.max_route,
                "map_return_after_game_over": self.return_after_game_over}

    # ---- per-level state --------------------------------------------------------
    def clear(self, new_level=True):
        """Forget the map. Called at every level change (a new level is a new
        puzzle) and when the sticky mask grows. NOT called on game over: the
        level restarts but its map is still true."""
        self.edges = {}        # key -> {action: next_key or DEAD}
        self.tries = {}        # key -> actions taken from this screen
        self.buttons = {}      # key -> frozenset of available button indices (0-4)
        self.clickable = {}    # key -> ACTION6 available on this screen
        self.cur = None        # key of the screen Goose is looking at now
        self.last = None       # (key, action) of the previous decision
        self.route = []        # remaining [(action, expected_next_key)]
        self.expect = None     # where the last route step should land
        self.arrived = False   # route just finished: try something untried here
        self.since_new = 0     # decisions since a never-seen screen
        self.found_at = {}     # key -> step at which the screen was first seen
        self.return_pending = False
        if new_level:
            self.stats["levels"] += 1
            self.attempt_steps = 0

    # ---- screen identity ----------------------------------------------------------
    def key(self, raw, live_mask):
        """Key of a raw (64, 64) frame under the sticky decoration mask."""
        if live_mask is not None and live_mask is not self._mask_seen:
            self._mask_seen = live_mask
            if (live_mask & ~self.sticky).any():
                self.sticky |= live_mask
                self.stats["mask_resets"] += 1
                self.clear(new_level=False)
        g = np.array(raw, dtype=np.uint8, copy=True)
        g[self.sticky] = 0
        return xxhash.xxh64(g.tobytes()).intdigest()

    @staticmethod
    def availability(available_actions):
        """(buttons 0-4 as a frozenset, click allowed) from the frame's list."""
        if not available_actions:
            return frozenset(range(5)), True
        vals = [getattr(a, "value", a) for a in available_actions]
        return frozenset(v - 1 for v in vals if 1 <= v <= 5), 6 in vals

    # ---- the three calls the agent makes each step ---------------------------------
    def observe(self, raw, live_mask, has_prev, available_actions):
        """Record the screen Goose is now looking at. `has_prev` is False right
        after a reset, a level change or a skipped frame (no edge to record)."""
        k = self.key(raw, live_mask)
        self.step += 1
        self.attempt_steps += 1
        new = k not in self.edges
        if new and len(self.edges) >= self.max_nodes:
            # The map is full, so this screen cannot be remembered - and then
            # it cannot be told from one seen a moment ago. Calling it new
            # every time (as before) kept resetting the stall counter, which
            # switched stall routing off for the rest of the level.
            new = False
        if new:
            self.edges[k] = {}
            self.tries[k] = 0
            self.found_at[k] = self.step
        if k in self.edges:
            self.buttons[k], self.clickable[k] = self.availability(available_actions)

        if self.expect is not None:              # checking a route step
            if has_prev and k == self.expect:
                self.stats["route_steps"] += 1
            else:
                self._abort("mismatch" if has_prev else "no_prev")
            self.expect = None

        if has_prev and self.last is not None:
            s, a = self.last
            if s in self.edges and k in self.edges:
                self.edges[s][a] = k
        if not has_prev:
            self.last = None
            if self.route or self.arrived:
                self._abort("no_prev")

        self.since_new = 0 if new else self.since_new + 1
        self.cur = k
        return k

    def choose(self):
        """A unified action index the map wants taken now, or None to let Goose
        choose. 0-4 = ACTION1-5, 5 + (64*y + x) = click."""
        k = self.cur
        if k is None:
            return None
        if self.route:                                   # 1. keep walking
            a, nxt = self.route.pop(0)
            if not self._allowed(k, a):
                self._abort("unavailable")
                return None
            self.expect = nxt
            if not self.route:
                self.arrived = True
            return a
        if self.arrived:                                 # 2. arrived: try something new
            self.arrived = False
            self.since_new = 0
            self.stats["routes_completed"] += 1
            untried = sorted(self._untried_buttons(k))
            if untried:
                self.stats["frontier_button_tries"] += 1
                return int(self.rng.choice(untried))
            return None
        if self.return_pending:                          # 3. back after a game over
            self.return_pending = False
            path = self._return_route(k)
            if path:
                self.stats["return_routes"] += 1
                return self._start(path)
        if self.since_new >= self.stall:                 # 4. stuck: nearest frontier
            self.since_new = 0
            path = self._bfs(k, self._is_frontier)
            if path is None:
                self.stats["no_frontier"] += 1
                return None
            self.stats["stall_routes"] += 1
            if not path:                                 # already standing on one
                self.arrived = True
                self.stats["routes_started"] += 1
                return self.choose()
            return self._start(path)
        return None

    def record(self, action):
        """The action finally taken from the current screen (Goose's or ours)."""
        if self.cur is None:
            return
        self.last = (self.cur, int(action))
        if self.cur in self.tries:
            self.tries[self.cur] += 1

    def on_game_over(self):
        """The previous move was followed by a game over."""
        self.stats["game_overs"] += 1
        self.attempts.append(self.attempt_steps)
        self.attempt_steps = 0
        if self.last is not None:
            s, a = self.last
            if s in self.edges and a not in self.edges[s]:
                self.edges[s][a] = DEAD
        self.last = None
        self.expect = None
        if self.route or self.arrived:
            self._abort("game_over")
        self.return_pending = self.return_after_game_over

    def return_limit(self):
        """Longest walk back allowed: half a typical attempt, capped at max_route."""
        if not self.attempts:
            return self.max_route
        return min(self.max_route, int(np.median(self.attempts)) // 2)

    # ---- convenience for the agent's sampler -----------------------------------------
    @staticmethod
    def unpack(a):
        """Unified index -> (action_idx, (y, x) or None, coord_idx or None),
        the triple Goose's sampler returns."""
        if a < 5:
            return a, None, None
        c = a - 5
        return 5, (c // GRID, c % GRID), c

    def override(self, action_idx, coords, coord_idx):
        a = self.choose()
        if a is None:
            return action_idx, coords, coord_idx
        self.stats["map_actions"] += 1
        return self.unpack(a)

    def summary(self):
        return dict(self.stats, nodes=len(self.edges), sticky_cells=int(self.sticky.sum()),
                    edges=sum(len(v) for v in self.edges.values()),
                    return_limit=self.return_limit())

    def dump(self, path):
        try:
            with open(path, "w") as f:
                json.dump(dict(self.summary(), **self.config()), f, indent=2)
        except OSError:
            pass

    # ---- internals -------------------------------------------------------------------
    def _allowed(self, k, a):
        if a < 5:
            return a in self.buttons.get(k, frozenset())
        return self.clickable.get(k, False)

    def _untried_buttons(self, k):
        return set(self.buttons.get(k, ())) - set(self.edges.get(k, {}))

    def _is_frontier(self, k):
        if self._untried_buttons(k):
            return True
        return self.clickable.get(k, False) and self.tries.get(k, 0) < self.click_tries

    def _bfs(self, start, goal):
        """Shortest known route [(action, next_key), ...] from `start` to the
        nearest screen satisfying `goal`; [] if `start` already does; None if
        nothing within `max_route` moves. Never uses a move marked DEAD."""
        if start not in self.edges:
            return None
        if goal(start):
            return []
        parent = {start: None}
        q = deque([(start, 0)])
        while q:
            s, d = q.popleft()
            if d >= self.max_route:
                continue
            for a, n in self.edges[s].items():
                if n == DEAD or n in parent or n not in self.edges:
                    continue
                parent[n] = (s, a)
                if goal(n):
                    return self._path(parent, n)
                q.append((n, d + 1))
        return None

    def _return_route(self, start):
        """Route to the most recently discovered screen that still has something
        untried, among those within return_limit() moves of `start`."""
        limit = self.return_limit()
        if start not in self.edges or limit < 1:
            return None
        parent = {start: None}
        q = deque([(start, 0)])
        best, best_t = None, -1
        while q:
            s, d = q.popleft()
            if d >= limit:
                continue
            for a, n in self.edges[s].items():
                if n == DEAD or n in parent or n not in self.edges:
                    continue
                parent[n] = (s, a)
                q.append((n, d + 1))
                t = self.found_at.get(n, -1)
                if t > best_t and self._is_frontier(n):
                    best, best_t = n, t
        return None if best is None else self._path(parent, best)

    @staticmethod
    def _path(parent, n):
        path = []
        while parent[n] is not None:
            p, pa = parent[n]
            path.append((pa, n))
            n = p
        return path[::-1]

    def _start(self, path):
        self.stats["routes_started"] += 1
        self.stats["route_len_total"] += len(path)
        self.route = list(path)
        return self.choose()

    def _abort(self, reason):
        self.stats["routes_aborted"] += 1
        self.stats["aborted_" + reason] += 1
        self.route = []
        self.expect = None
        self.arrived = False
