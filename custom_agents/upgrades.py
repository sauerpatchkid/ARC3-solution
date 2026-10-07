"""upgrades.py — improvements to the novelty label and the return map, each
behind EVAL_UPGRADES, kept separate from both.

Nothing here runs unless EVAL_UPGRADES is set; with it unset the agent is
exactly the novelty-label agent (or the return-map agent). The ADOPTED Goose
(mb_gated_att, docs/plans/upgrade-confirm-results.md) is

    EVAL_LABEL=novel EVAL_RETURN_MAP=1 EVAL_UPGRADES=bars,map_gated,attempt

which custom_agents/presets.py registers as the agent `mb_gated_att`.

Label upgrades (change what Goose trains on; used on top of EVAL_LABEL=novel):
  bars       Decoration mask that also catches progress bars that move at the
             same time as real changes (component-level, see BarDetector).
             The shared detector misses these on most games; measured on the
             novelty-label corpora it masks nothing on 12 of 14 games while
             this finds a bar on 10 of them. Applies to the novelty fingerprint
             and, when the map is on, to the map's fingerprint too.
  attempt    Reward a move 1.0 for a screen new this level, 0.5 for a screen new
             this ATTEMPT (since the last game over) but seen before, else 0.
             Meant to be used with `bars` (without it every screen in an
             attempt looks new because the move-count bar changes).

Sampling upgrade (on top of the novelty label):
  deadclick  A click on the same cell and colour that has done nothing 4 times
             this level is never chosen again this level (Goose resamples).
             Not part of the adopted agent; kept because Rulebook v2's LLM-free
             arm builds on it (docs/plans/rulebook-v2.md, C2-free).

Map upgrade (needs EVAL_RETURN_MAP=1; changes custom_agents/return_map.py's
behaviour through a subclass, never by editing it):
  map_gated    Walk back after a game over only while walking back pays: a
               two-armed sliding-window UCB bandit per game ("walk back" vs
               "stay") scored by new screens per move in each attempt.

REMOVED 2026-10-06: `graded`, `map_objects` and `map_diverse` lost the upgrade
screen (docs/plans/upgrade-screen-results.md) and were deleted. Their code and
tests are at the git tags `upgrade-screen-round1` and `pre-cleanup-2026-10-06`.
The grid helpers they used (label_components, screen_objects) live on in
custom_agents/gridtools.py.

HOW TO REMOVE IT
Every hook line in custom_agents/action.py, check_repo.py and the root
Makefile ends in `# [upgrades]`; `sed -i '/\\[upgrades\\]/d' <file>` restores
each byte for byte. Then delete this file, tests/test_upgrades.py and
experiments/upgrade_screen/. Neither the novelty label nor the return map
depends on anything here.
"""
import json
import os
from collections import deque

import numpy as np
import xxhash

from gridtools import GRID, BAR_BORDER, border_band, tick_cells
from return_map import ReturnMap

LABEL_OPTIONS = {"bars", "attempt"}
MAP_OPTIONS = {"map_gated"}
KNOWN = LABEL_OPTIONS | MAP_OPTIONS | {"deadclick"}
REMOVED = {"graded", "map_objects", "map_diverse"}

DEAD_CLICK_TRIES = 4        # BDR-Pro's threshold: >= 4 no-op clicks on one appearance
ATTEMPT_REWARD = 0.5        # reward for "new this attempt, seen before this level"
GATE_WINDOW = 50            # sliding window (attempts) per bandit arm
GATE_MIN_TRIES = 3          # try each arm this often before trusting the bandit


# --------------------------------------------------------------------------------
# Decorations: progress bars that tick at the same time as real changes
# --------------------------------------------------------------------------------
class BarDetector:
    """Rotating-ticker detection at the connected-component level, with two
    guards measured on the corpora: (1) only cells in the outer 4-cell border
    band count (unguarded, it flagged mid-screen playfield rows on sk48, s5i5
    and tr87; every real bar found sat on rows 0-3 or 60-63 or column 0), and
    (2) the mask is sticky (a cell once flagged stays flagged), because the
    live masks flip back and forth. Thresholds mirror metrics_common: at least
    30% of moves contain a tick, and the mask covers 95% of tick changes."""

    def __init__(self, warmup=200, refresh=250, min_tick_frac=0.30,
                 coverage=0.95, max_cells=256):
        self.warmup, self.refresh = warmup, refresh
        self.min_tick_frac, self.coverage, self.max_cells = min_tick_frac, coverage, max_cells
        self.n = 0
        self.n_tick = 0
        self.tick_count = np.zeros((GRID, GRID), dtype=np.int64)
        self.sticky = np.zeros((GRID, GRID), dtype=bool)
        self.band = border_band(BAR_BORDER)
        self.growths = 0

    def update(self, prev_raw, cur_raw):
        diff = prev_raw != cur_raw
        self.n += 1
        if diff.any():
            t = tick_cells(diff)
            if t.any():
                self.n_tick += 1
                self.tick_count += t
        if self.n >= self.warmup and (self.n == self.warmup or self.n % self.refresh == 0):
            self._recompute()

    def _recompute(self):
        if self.n_tick / self.n < self.min_tick_frac:
            return
        counts = self.tick_count.ravel().astype(float)
        total = counts.sum()
        if total <= 0:
            return
        order = np.argsort(-counts)
        k = int(np.searchsorted(np.cumsum(counts[order]) / total, self.coverage)) + 1
        if k > self.max_cells:
            return                                   # too diffuse to be a bar
        rot = np.zeros(GRID * GRID, dtype=bool)
        rot[order[:k]] = True
        rot = rot.reshape(GRID, GRID) & self.band
        if (rot & ~self.sticky).any():
            self.sticky |= rot
            self.growths += 1


# --------------------------------------------------------------------------------
# The return map, upgraded (a subclass: return_map.py itself is never edited)
# --------------------------------------------------------------------------------
class UpgradedReturnMap(ReturnMap):
    def __init__(self, *args, gated=False, bars=None, **kw):
        super().__init__(*args, **kw)
        self.gated, self.bars = gated, bars
        # bandit state is per GAME: it survives level changes
        self.gate_rewards = {"walk": deque(maxlen=GATE_WINDOW), "stay": deque(maxlen=GATE_WINDOW)}
        self.gate_arm = "stay"          # the first attempt never walks back
        self.attempt_new = 0
        self.stats.update({"gate_walk": 0, "gate_stay": 0})

    def clear(self, new_level=True):
        super().clear(new_level)
        self.attempt_new = 0    # new screens this attempt (for map_gated)

    # -- fingerprint: add the bar mask ----------------------------------------------
    def key(self, raw, live_mask):
        if self.bars is not None and live_mask is not None:
            live_mask = live_mask | self.bars.sticky
        return super().key(raw, live_mask)

    def observe(self, raw, live_mask, has_prev, available_actions):
        before = len(self.edges)
        k = super().observe(raw, live_mask, has_prev, available_actions)
        if len(self.edges) > before:
            self.attempt_new += 1
        return k

    # -- map_gated: walk back only while it pays --------------------------------------
    # KNOWN QUIRK, left as confirmed: a level change does not reset gate_arm, so
    # a new level's first attempt (which never walks back, and finds many new
    # screens) is credited to whichever arm was active before the level-up.
    # The 50-attempt window washes it out. Changing it changes the adopted
    # agent, so it needs its own test and confirm.
    def on_game_over(self):
        moves = max(self.attempt_steps, 1)
        self.gate_rewards[self.gate_arm].append(self.attempt_new / moves)
        self.attempt_new = 0
        super().on_game_over()
        if self.gated:
            self.gate_arm = self._gate_choice()
            self.stats["gate_" + self.gate_arm] += 1
            self.return_pending = self.return_pending and self.gate_arm == "walk"

    def _gate_choice(self):
        """Sliding-window UCB over {walk, stay} (Agent57's meta-controller idea,
        at the scale of one decision per attempt)."""
        for arm in ("walk", "stay"):
            if len(self.gate_rewards[arm]) < GATE_MIN_TRIES:
                return arm
        total = sum(len(v) for v in self.gate_rewards.values())
        means = {a: float(np.mean(v)) for a, v in self.gate_rewards.items()}
        scale = max(max(means.values()), 1e-3)
        ucb = {a: means[a] + scale * np.sqrt(2 * np.log(total) / len(self.gate_rewards[a]))
               for a in means}
        return max(ucb, key=ucb.get)

    def summary(self):
        s = super().summary()
        for arm, v in self.gate_rewards.items():
            if v:                                   # only real numbers: TensorBoard logs these
                s["gate_mean_" + arm] = float(np.mean(v))
        return s


# --------------------------------------------------------------------------------
# The upgrades object the agent talks to
# --------------------------------------------------------------------------------
class Upgrades:
    def __init__(self, names, rng=None):
        names = {n.strip() for n in names if n.strip()}
        gone = names & REMOVED
        if gone:
            raise SystemExit(f"EVAL_UPGRADES: {sorted(gone)} lost the upgrade screen and were removed "
                             f"on 2026-10-06; run them from git tag upgrade-screen-round1")
        unknown = names - KNOWN
        if unknown:
            raise SystemExit(f"EVAL_UPGRADES: unknown {sorted(unknown)}; known {sorted(KNOWN)}")
        self.names = names
        self.rng = rng if rng is not None else np.random
        self.bars = BarDetector() if "bars" in names else None
        self.label_on = bool(names & LABEL_OPTIONS)
        self.visits = {}
        self.seen_attempt = set()
        self.label = None
        self.prev_key = None
        self.dead = {}          # (y, x, colour) -> no-op clicks this level
        self.blocked = set()
        self.stats = {"dead_click_resamples": 0, "blocked_contexts": 0,
                      "attempt_rewards": 0, "bar_growths": 0}

    @classmethod
    def from_env(cls):
        return cls(os.getenv("EVAL_UPGRADES", "").split(","))

    def check(self, map_on):
        if self.names & MAP_OPTIONS and not map_on:
            raise SystemExit(f"EVAL_UPGRADES {sorted(self.names & MAP_OPTIONS)} need EVAL_RETURN_MAP=1")

    def config(self):
        return {"upgrades": ",".join(sorted(self.names))}

    def wrap_map(self, m):
        """Replace the return map with the upgraded subclass (same settings)."""
        if not (self.names & MAP_OPTIONS) and self.bars is None:
            return m
        return UpgradedReturnMap(stall=m.stall, click_tries=m.click_tries, max_route=m.max_route,
                                 return_after_game_over=m.return_after_game_over,
                                 max_nodes=m.max_nodes, rng=m.rng,
                                 gated="map_gated" in self.names, bars=self.bars)

    # ---- per step -----------------------------------------------------------------
    def key(self, raw, shared_mask):
        m = shared_mask if self.bars is None else (shared_mask | self.bars.sticky)
        g = np.array(raw, dtype=np.uint8, copy=True)
        g[m] = 0
        return xxhash.xxh64(g.tobytes()).intdigest()

    def observe(self, prev_raw, cur_raw, shared_mask, prev_action):
        """Every decision frame. prev_raw is None right after a reset / level
        change / skipped frame (no transition to learn from)."""
        if self.bars is not None and prev_raw is not None:
            before = self.bars.growths
            self.bars.update(prev_raw, cur_raw)
            self.stats["bar_growths"] += self.bars.growths - before
        k = self.key(cur_raw, shared_mask)
        self.label = None
        if prev_raw is not None:
            v = self.visits.get(k, 0)
            if "attempt" in self.names:
                if v == 0:
                    self.label = 1.0
                elif k not in self.seen_attempt:
                    self.label = ATTEMPT_REWARD
                    self.stats["attempt_rewards"] += 1
                else:
                    self.label = 0.0
            elif self.label_on:
                self.label = 1.0 if v == 0 else 0.0
            if ("deadclick" in self.names and prev_action is not None and prev_action >= 5
                    and self.prev_key == k):
                y, x = divmod(prev_action - 5, GRID)
                ctx = (y, x, int(prev_raw[y, x]))
                self.dead[ctx] = self.dead.get(ctx, 0) + 1
                if self.dead[ctx] == DEAD_CLICK_TRIES:
                    self.blocked.add(ctx)
                    self.stats["blocked_contexts"] += 1
        self.visits[k] = self.visits.get(k, 0) + 1
        self.seen_attempt.add(k)
        self.prev_key = k

    def reward(self, default):
        """The training label: the upgraded one if a label option is on."""
        return default if self.label is None else float(self.label)

    def override(self, action_idx, coords, coord_idx, all_probs, cur_raw):
        """deadclick: if Goose picked a click proven dead on this appearance,
        resample from its own distribution with every dead click removed."""
        if "deadclick" not in self.names or action_idx != 5 or not self.blocked:
            return action_idx, coords, coord_idx
        y, x = coords
        if (y, x, int(cur_raw[y, x])) not in self.blocked:
            return action_idx, coords, coord_idx
        p = np.asarray(all_probs, dtype=np.float64).copy()
        p[5:] /= GRID * GRID
        b = np.array(list(self.blocked), dtype=np.int64)
        live = cur_raw[b[:, 0], b[:, 1]] == b[:, 2]
        p[5 + b[live, 0] * GRID + b[live, 1]] = 0.0
        if not np.isfinite(p).all() or p.sum() <= 0:
            return action_idx, coords, coord_idx
        self.stats["dead_click_resamples"] += 1
        return ReturnMap.unpack(int(self.rng.choice(len(p), p=p / p.sum())))

    def on_game_over(self):
        self.seen_attempt = set()
        self.prev_key = None

    def on_level(self):
        self.visits = {}
        self.seen_attempt = set()
        self.dead = {}
        self.blocked = set()
        self.prev_key = None

    def summary(self):
        return dict(self.stats, bar_cells=int(self.bars.sticky.sum()) if self.bars is not None else 0)

    def dump(self, path, return_map=None):
        out = dict(self.summary(), **self.config())
        try:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
        except OSError:
            pass
