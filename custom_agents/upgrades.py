"""upgrades.py — the upgrade screen: candidate improvements to the novelty label
and the return map, each behind EVAL_UPGRADES, kept separate from both.

This module is EXPERIMENTAL and self-contained. It exists so a short screening
sweep (experiments/upgrade_screen/) can try several ideas cheaply and rank
them. Nothing here runs unless EVAL_UPGRADES is set; with it unset the agent is
exactly the adopted novelty-label agent (or the return-map agent).

    EVAL_UPGRADES=bars,attempt      # comma-separated; names below

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
  graded     Reward 1/sqrt(visits): 1.0 the first time a screen is reached this
             level, 0.71 the second, 0.58 the third... instead of 1-or-0.

Sampling upgrade (on top of the novelty label):
  deadclick  A click on the same cell and colour that has done nothing 4 times
             this level is never chosen again this level (Goose resamples).

Map upgrades (need EVAL_RETURN_MAP=1; change custom_agents/return_map.py's
behaviour through a subclass, never by editing it):
  map_gated    Walk back after a game over only while walking back pays: a
               two-armed sliding-window UCB bandit per game ("walk back" vs
               "stay") scored by new screens per move in each attempt.
  map_objects  On click screens, "untried" means an object never clicked there
               (single-colour connected components, most salient first), and
               arriving at such a screen clicks the next untried object.
  map_diverse  Choose the walk-back target at random, weighted 1/sqrt(visits+1)
               (Go-Explore), instead of always the most recent discovery.

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

from return_map import ReturnMap, DEAD

GRID = 64
LABEL_OPTIONS = {"bars", "attempt", "graded"}
MAP_OPTIONS = {"map_gated", "map_objects", "map_diverse"}
KNOWN = LABEL_OPTIONS | MAP_OPTIONS | {"deadclick"}

DEAD_CLICK_TRIES = 4        # BDR-Pro's threshold: >= 4 no-op clicks on one appearance
ATTEMPT_REWARD = 0.5        # reward for "new this attempt, seen before this level"
BAR_BORDER = 4              # bar cells must lie within this many cells of an edge
GATE_WINDOW = 50            # sliding window (attempts) per bandit arm
GATE_MIN_TRIES = 3          # try each arm this often before trusting the bandit
MAX_OBJECTS = 64
MAX_OBJECT_CELLS = 1024     # bigger single-colour regions are background, not objects


# --------------------------------------------------------------------------------
# Decorations: progress bars that tick at the same time as real changes
# --------------------------------------------------------------------------------
_SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _neighbour_sum(m):
    """For each cell, how many of its 8 neighbours are True (numpy, no scipy)."""
    p = np.pad(m.astype(np.int16), 1)
    return sum(p[1 + dy:1 + dy + GRID, 1 + dx:1 + dx + GRID] for dy, dx in _SHIFTS)


def tick_cells(diff):
    """Changed cells that belong to an 8-connected changed component of at most
    2 cells. The shared detector only sees a tick when the WHOLE move changed
    <= 2 cells; this sees a small tick inside a big move (legacy/llm_track/
    tickers.py's idea, vectorised). Size 1: no changed neighbour. Size 2: one
    changed neighbour whose only changed neighbour is this cell."""
    nb = _neighbour_sum(diff)
    single = diff & (nb == 0)
    one = diff & (nb == 1)
    pair = one & (_neighbour_sum(one) == 1)
    return single | pair


def border_band(width=BAR_BORDER):
    band = np.zeros((GRID, GRID), dtype=bool)
    band[:width, :] = band[-width:, :] = True
    band[:, :width] = band[:, -width:] = True
    return band


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
        self.band = border_band()
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
# Objects on a click screen (single-colour connected components, numpy only)
# --------------------------------------------------------------------------------
def label_components(grid, valid):
    """4-connected single-colour components of `grid` among `valid` cells.
    Min-label propagation with pointer jumping; returns int32 labels (-1 where
    not valid), each label the smallest flat index in its component."""
    lab = np.where(valid, np.arange(GRID * GRID, dtype=np.int32).reshape(GRID, GRID), -1)
    big = np.int32(GRID * GRID)
    right = valid[:, :-1] & valid[:, 1:] & (grid[:, :-1] == grid[:, 1:])
    down = valid[:-1, :] & valid[1:, :] & (grid[:-1, :] == grid[1:, :])
    work = np.where(valid, lab, big)
    for _ in range(GRID * 2):
        new = work.copy()
        new[:, :-1] = np.where(right, np.minimum(new[:, :-1], work[:, 1:]), new[:, :-1])
        new[:, 1:] = np.where(right, np.minimum(new[:, 1:], work[:, :-1]), new[:, 1:])
        new[:-1, :] = np.where(down, np.minimum(new[:-1, :], work[1:, :]), new[:-1, :])
        new[1:, :] = np.where(down, np.minimum(new[1:, :], work[:-1, :]), new[1:, :])
        flat = new.ravel()
        v = flat < big
        flat[v] = np.minimum(flat[v], flat[flat[v]])       # pointer jumping
        new = flat.reshape(GRID, GRID)
        if np.array_equal(new, work):
            break
        work = new
    return np.where(valid, work, -1).astype(np.int32)


def screen_objects(raw, masked):
    """Objects worth clicking on this screen, most salient first, as
    (labels, [(label, target_flat_index), ...]). Background (the most common
    colour), masked decoration cells and regions over MAX_OBJECT_CELLS are
    excluded. Salience follows the 3rd-place ARC-AGI-3 agent's idea (size and
    colour rarity): small objects in rare colours first. The target is the
    object's cell nearest its centroid."""
    colours, counts = np.unique(raw[~masked], return_counts=True)
    if len(colours) == 0:
        return np.full((GRID, GRID), -1, np.int32), []
    background = colours[np.argmax(counts)]
    rarity = np.ones(16)
    rarity[colours] = counts / counts.sum()
    valid = (~masked) & (raw != background)
    labels = label_components(raw, valid)
    ys, xs = np.nonzero(labels >= 0)
    if len(ys) == 0:
        return labels, []
    ids, inv, sizes = np.unique(labels[ys, xs], return_inverse=True, return_counts=True)
    cy = np.bincount(inv, weights=ys) / sizes
    cx = np.bincount(inv, weights=xs) / sizes
    d = (ys - cy[inv]) ** 2 + (xs - cx[inv]) ** 2
    order = np.lexsort((d, inv))                      # by object, then distance
    _, first = np.unique(inv[order], return_index=True)
    near = order[first]                               # each object's most central cell
    colour = raw[ys[near], xs[near]].astype(np.int64)
    sal = sizes * rarity[colour]
    keep = sizes <= MAX_OBJECT_CELLS
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(sal[idx], kind="stable")][:MAX_OBJECTS]
    return labels, [(int(ids[i]), int(ys[near[i]]) * GRID + int(xs[near[i]])) for i in idx]


# --------------------------------------------------------------------------------
# The return map, upgraded (a subclass: return_map.py itself is never edited)
# --------------------------------------------------------------------------------
class UpgradedReturnMap(ReturnMap):
    def __init__(self, *args, gated=False, objects=False, diverse=False, bars=None, **kw):
        super().__init__(*args, **kw)
        self.gated, self.objects, self.diverse, self.bars = gated, objects, diverse, bars
        # bandit state is per GAME: it survives level changes
        self.gate_rewards = {"walk": deque(maxlen=GATE_WINDOW), "stay": deque(maxlen=GATE_WINDOW)}
        self.gate_arm = "stay"          # the first attempt never walks back
        self.attempt_new = 0
        self.stats.update({"gate_walk": 0, "gate_stay": 0, "object_clicks": 0})
        self._labels = None

    def clear(self, new_level=True):
        super().clear(new_level)
        self.obj_targets = {}   # key -> [(label, target_index), ...]
        self.obj_tried = {}     # key -> set of labels clicked from that screen
        self.visits = {}        # key -> times observed (for map_diverse)
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
        self.visits[k] = self.visits.get(k, 0) + 1
        self._labels = None
        if self.objects and self.clickable.get(k, False):
            self._labels, targets = screen_objects(np.asarray(raw), self.sticky)
            if k not in self.obj_targets:
                self.obj_targets[k] = targets
                self.obj_tried[k] = set()
        return k

    def record(self, action):
        super().record(action)
        if self.objects and action >= 5 and self._labels is not None and self.cur in self.obj_tried:
            y, x = divmod(action - 5, GRID)
            lid = int(self._labels[y, x])
            if lid >= 0:
                self.obj_tried[self.cur].add(lid)

    # -- map_objects: what counts as untried, and what to click on arrival -----------
    def _untried_objects(self, k):
        tried = self.obj_tried.get(k, set())
        return [tgt for lid, tgt in self.obj_targets.get(k, []) if lid not in tried]

    def _is_frontier(self, k):
        if self.objects and k in self.obj_targets and self.clickable.get(k, False):
            return bool(self._untried_buttons(k)) or bool(self._untried_objects(k))
        return super()._is_frontier(k)

    def choose(self):
        arriving = self.arrived and not self.route
        a = super().choose()
        if a is None and arriving and self.objects and self.clickable.get(self.cur, False):
            left = self._untried_objects(self.cur)
            if left:
                self.stats["object_clicks"] += 1
                return 5 + left[0]
        return a

    # -- map_diverse: Go-Explore-style target choice ---------------------------------
    def _return_route(self, start):
        if not self.diverse:
            return super()._return_route(start)
        limit = self.return_limit()
        if start not in self.edges or limit < 1:
            return None
        parent = {start: None}
        q = deque([(start, 0)])
        cands = []
        while q:
            s, d = q.popleft()
            if d >= limit:
                continue
            for a, n in self.edges[s].items():
                if n == DEAD or n in parent or n not in self.edges:
                    continue
                parent[n] = (s, a)
                q.append((n, d + 1))
                if self._is_frontier(n):
                    cands.append(n)
        if not cands:
            return None
        w = np.array([1.0 / np.sqrt(self.visits.get(n, 0) + 1) for n in cands])
        pick = cands[int(self.rng.choice(len(cands), p=w / w.sum()))]
        return self._path(parent, pick)

    # -- map_gated: walk back only while it pays --------------------------------------
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
                                 gated="map_gated" in self.names,
                                 objects="map_objects" in self.names,
                                 diverse="map_diverse" in self.names, bars=self.bars)

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
            elif "graded" in self.names:
                self.label = 1.0 / np.sqrt(v + 1)
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
