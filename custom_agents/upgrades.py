"""upgrades.py — improvements to the novelty label and the return map, each
behind EVAL_UPGRADES, kept separate from both.

Nothing here runs unless EVAL_UPGRADES is set; with it unset the agent is
exactly the novelty-label agent (or the return-map agent). The ADOPTED Goose
(mb_gated_att, docs/1-goose-upgrades/upgrade-confirm-results.md) is

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

Sampling upgrades (on top of the novelty label):
  deadclick  A click on the same cell and colour that has done nothing 4 times
             this level is never chosen again this level (Goose resamples).
             Not part of the adopted agent; kept because Rulebook v2's LLM-free
             arm builds on it (docs/2-llm-history/rulebook-v2.md, C2-free).
  objclick   Where a click lands is chosen by shape, not by cell (see ObjClick).
             Goose picks a click cell in proportion to the click head's output,
             so a large background takes most clicks and small objects are found
             slowly. With probability 0.9 the cell is drawn again with weight =
             its click probability / the size of its shape; otherwise Goose's
             cell is kept, so every cell stays reachable. Whether the move is a
             button or a click is never changed. Not part of the adopted agent
             (upgrade screen 3, docs/1-goose-upgrades/upgrade-screen3.md).

Map upgrade (needs EVAL_RETURN_MAP=1; changes custom_agents/return_map.py's
behaviour through a subclass, never by editing it):
  map_gated    Walk back after a game over only while walking back pays: a
               two-armed sliding-window UCB bandit per game ("walk back" vs
               "stay") scored by new screens per move in each attempt.
  map_sched    A bandit over five map SCHEDULES instead of two (see SCHEDULES):
               how long Goose may go without a new screen before the map steps
               in (200 or 30 decisions), how much of a typical attempt a walk
               back may use (none, 50%, 80%) and whether a button never pressed
               on the current screen is pressed before Goose's own pick. It
               takes map_gated's place (the two together are refused). A choice
               is made at every game over and after 100 decisions without one.
               Not part of the adopted agent (upgrade screen 3).

Carry upgrades (what one level teaches about the next; no LLM; the code is in
custom_agents/carry.py, which says where it comes from and what is known):
  carry_noops  A colour clicked 20 times or more on a level, with no click ever
               changing a live cell, is carried as a no-op: on later levels a
               click on it keeps a tenth of its probability, until one changes
               something.
  carry_hints  The two colours that vanished or shrank most before a level was
               won become goal hints on the next level: a click on one counts
               double, until 2,000 moves pass without it changing.
  Not part of the adopted agent. Their test: docs/1-goose-upgrades/carry-inside-goose.md.

REMOVED 2026-10-06: `graded`, `map_objects` and `map_diverse` lost the upgrade
screen (docs/1-goose-upgrades/upgrade-screen-results.md) and were deleted. Their code and
tests are at the git tags `upgrade-screen-round1` and `pre-cleanup-2026-10-06`.
The grid helpers they used (label_components, screen_objects) live on in
custom_agents/gridtools.py.

HOW TO REMOVE IT
Every hook line in custom_agents/action.py, check_repo.py and the root
Makefile ends in `# [upgrades]`; `sed -i '/\\[upgrades\\]/d' <file>` restores
each byte for byte. Then delete this file, tests/test_upgrades.py and
legacy/goose_experiments/upgrade_screen/. Neither the novelty label nor the return map
depends on anything here.
"""
import json
import os
from collections import deque

import numpy as np
import xxhash

from carry import Carry
from gridtools import GRID, BAR_BORDER, border_band, label_components, tick_cells
from return_map import ReturnMap

LABEL_OPTIONS = {"bars", "attempt"}
MAP_OPTIONS = {"map_gated", "map_sched"}
CARRY_OPTIONS = {"carry_noops", "carry_hints"}
KNOWN = LABEL_OPTIONS | MAP_OPTIONS | CARRY_OPTIONS | {"deadclick", "objclick"}
REMOVED = {"graded", "map_objects", "map_diverse"}

DEAD_CLICK_TRIES = 4        # BDR-Pro's threshold: >= 4 no-op clicks on one appearance
ATTEMPT_REWARD = 0.5        # reward for "new this attempt, seen before this level"
GATE_WINDOW = 50            # sliding window (attempts) per bandit arm
GATE_MIN_TRIES = 3          # try each arm this often before trusting the bandit

# map_sched: schedule -> (stall, the share of a typical attempt a walk back may use, untried button first).
#   stall: decisions without a new screen before the map routes to the nearest frontier
#   share: 0 = no walk back after a game over; 0.5 is the plain map's limit
#   untried button first: a button never pressed on the current screen goes before Goose's pick
SCHEDULES = {
    "stay200": (200, 0.0, False),       # what map_gated calls "stay"
    "walk200": (200, 0.5, False),       # what map_gated calls "walk"
    "stay30": (30, 0.0, False),
    "walk30": (30, 0.5, False),
    "sweep30": (30, 0.8, True),
}
SCHED_START = "stay200"     # the schedule a game starts on
SCHED_WINDOW = 50           # sliding window (pulls) per schedule
SCHED_MIN_TRIES = 2         # try each schedule this often before trusting the bandit
SCHED_STRETCH = 100         # decisions without a game over before a new choice is made

OBJCLICK_REDRAW = 0.9       # share of Goose's clicks drawn again by shape; the rest keep Goose's cell
OBJCLICK_SEED = 104729      # added to the agent's seed for objclick's own generator (as carry.py does)
OBJCLICK_CACHE = 256        # screens whose shapes are remembered


def ucb_choice(rewards):
    """Sliding-window UCB over the arms of `rewards` (name -> its last rewards,
    none empty): the arm with the highest mean + scale * sqrt(2 ln(total) / pulls),
    scale = the best arm's mean. Agent57's meta-controller idea, at the scale of
    one decision per attempt. map_gated and map_sched both choose with it."""
    total = sum(len(v) for v in rewards.values())
    means = {a: float(np.mean(v)) for a, v in rewards.items()}
    scale = max(max(means.values()), 1e-3)
    ucb = {a: means[a] + scale * np.sqrt(2 * np.log(total) / len(rewards[a]))
           for a in means}
    return max(ucb, key=ucb.get)


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
    def __init__(self, *args, gated=False, bars=None, sched=False, **kw):
        super().__init__(*args, **kw)
        self.gated, self.bars, self.sched = gated, bars, sched
        # bandit state is per GAME: it survives level changes
        self.gate_rewards = {"walk": deque(maxlen=GATE_WINDOW), "stay": deque(maxlen=GATE_WINDOW)}
        self.gate_arm = "stay"          # the first attempt never walks back
        self.attempt_new = 0
        self.stats.update({"gate_walk": 0, "gate_stay": 0})
        if sched:                       # map_sched: its own bandit, also per game
            self.sched_rewards = {name: deque(maxlen=SCHED_WINDOW) for name in SCHEDULES}
            self.sched_arm = SCHED_START
            self.stall = SCHEDULES[SCHED_START][0]
            for name in SCHEDULES:      # pulls, decisions made and new screens found under each schedule
                self.stats.update({"sched_pulls_" + name: 0, "sched_decisions_" + name: 0,
                                   "sched_new_" + name: 0})
            self.stats["sched_pulls_" + SCHED_START] = 1
            self.stats["untried_first"] = 0

    def clear(self, new_level=True):
        super().clear(new_level)
        self.attempt_new = 0    # new screens this attempt (for map_gated)
        self.sched_new = 0      # new screens, and decisions, in the stretch the current
        self.sched_steps = 0    # schedule is running (for map_sched): a new map starts both again

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
        if self.sched:
            self._sched_count(len(self.edges) > before)
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
        if self.sched:
            self._sched_switch()
            if not SCHEDULES[self.sched_arm][1]:    # this schedule does not walk back
                self.return_pending = False

    def _gate_choice(self):
        """Sliding-window UCB over {walk, stay} (see ucb_choice)."""
        for arm in ("walk", "stay"):
            if len(self.gate_rewards[arm]) < GATE_MIN_TRIES:
                return arm
        return ucb_choice(self.gate_rewards)

    # -- map_sched: a bandit over five schedules ---------------------------------------
    # A STRETCH is what one choice covers: from the choice to the next game over, or
    # SCHED_STRETCH decisions, whichever comes first. The schedule that ran it is
    # credited with new screens per decision, the measure map_gated uses.
    # KNOWN QUIRKS, shared with map_gated: a new level's first screens are all new and
    # are credited to whichever schedule is running; so are the screens found again
    # after the mask grows and the map starts over.
    def _sched_count(self, new):
        """One decision under the running schedule; `new`: it showed a new screen."""
        self.sched_steps += 1
        self.sched_new += int(new)
        self.stats["sched_decisions_" + self.sched_arm] += 1
        self.stats["sched_new_" + self.sched_arm] += int(new)
        if self.sched_steps >= SCHED_STRETCH:       # no game over for a long while: choose anyway
            self._sched_switch()

    def _sched_switch(self):
        """End the stretch: credit the schedule that ran it, then choose the next.
        A stretch with no decision in it (a game over straight after a choice)
        credits nothing and keeps the schedule."""
        if self.sched_steps:
            self.sched_rewards[self.sched_arm].append(self.sched_new / self.sched_steps)
            self.sched_arm = self._sched_choice()
            self.stats["sched_pulls_" + self.sched_arm] += 1
            self.stall = SCHEDULES[self.sched_arm][0]
        self.sched_steps = self.sched_new = 0

    def _sched_choice(self):
        """Each schedule SCHED_MIN_TRIES times in turn (the least tried first, in the
        table's order), then sliding-window UCB (see ucb_choice)."""
        least = min(SCHEDULES, key=lambda name: len(self.sched_rewards[name]))
        if len(self.sched_rewards[least]) < SCHED_MIN_TRIES:
            return least
        return ucb_choice(self.sched_rewards)

    def return_limit(self):
        """Longest walk back allowed. map_sched: the running schedule's share of a
        typical attempt, where the plain map allows half."""
        share = SCHEDULES[self.sched_arm][1] if self.sched else 0.5
        if share == 0.5 or not self.attempts:
            return super().return_limit()
        return min(self.max_route, int(np.median(self.attempts) * share))

    def choose(self):
        """The map's move, or None for Goose's. map_sched, on a schedule with
        "untried button first": when the map has nothing to say (no route step, no
        walk back, no stall route) and this screen has a button never pressed from
        it, press one of those at random. No move is ever forbidden: once every
        button has been pressed here, Goose picks as usual."""
        a = super().choose()
        if a is None and self.sched and SCHEDULES[self.sched_arm][2] and self.cur is not None:
            untried = sorted(self._untried_buttons(self.cur))
            if untried:
                self.stats["untried_first"] += 1
                return int(self.rng.choice(untried))
        return a

    def summary(self):
        s = super().summary()
        for arm, v in self.gate_rewards.items():
            if v:                                   # only real numbers: TensorBoard logs these
                s["gate_mean_" + arm] = float(np.mean(v))
        if self.sched:
            for name, v in self.sched_rewards.items():
                if v:                               # over its last SCHED_WINDOW pulls
                    s["sched_mean_" + name] = float(np.mean(v))
        return s


# --------------------------------------------------------------------------------
# objclick: where a click lands is chosen by shape, not by cell
# --------------------------------------------------------------------------------
def shape_sizes(raw, mask):
    """For each cell of a screen, the number of cells in its shape (flat, 4,096
    long), and the number of shapes on the screen. A shape is a 4-connected patch
    of one colour; the cells under the decoration mask count together as one."""
    valid = ~mask
    lab = label_components(raw, valid)
    counts = np.bincount(lab[valid], minlength=GRID * GRID)     # a label is a flat cell index
    size = np.where(valid, counts[np.maximum(lab, 0)], int(mask.sum()))
    return size.ravel(), int((counts > 0).sum()) + int(mask.any())


class ObjClick:
    """Goose draws a click cell in proportion to the click head's output, so a
    shape gets clicks in proportion to its SIZE times its mean score: a large
    background takes most of them. Here, with probability OBJCLICK_REDRAW, the
    cell is drawn again with weight = its click probability / the size of its
    shape. That is the same as choosing a shape in proportion to its mean score
    and then a cell inside it in proportion to its own score. Otherwise Goose's
    cell is kept, so every cell stays reachable (r11l is answered at computed
    spots). There is no fixed click order, and a button is never touched.

    It draws from its own generator, seeded from the agent's seed, so Goose's
    random stream is untouched. NOT the removed `map_objects`: that changed what
    the map called untried; this changes only where a click lands."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed + OBJCLICK_SEED)
        self.mask = None        # the decoration mask of this decision's screen (Upgrades.observe sets it)
        self.cache = {}         # screen -> shape_sizes(screen), the last OBJCLICK_CACHE screens
        self.stats = {"objclick_redrawn": 0, "objclick_kept": 0, "objclick_moved": 0,
                      "objclick_no_weights": 0, "objclick_screens": 0, "objclick_shapes": 0}

    def shapes(self, raw):
        mask = self.mask if self.mask is not None else np.zeros((GRID, GRID), dtype=bool)
        key = xxhash.xxh64(raw.tobytes() + np.packbits(mask).tobytes()).intdigest()
        found = self.cache.get(key)
        if found is None:
            if len(self.cache) >= OBJCLICK_CACHE:
                del self.cache[next(iter(self.cache))]          # the oldest
            found = self.cache[key] = shape_sizes(raw, mask)
            self.stats["objclick_screens"] += 1
        return found

    def override(self, action_idx, coords, coord_idx, all_probs, raw):
        """Goose's sampled move, with a click's cell drawn again by shape (class
        doc). Returns the triple Goose's sampler returns."""
        if action_idx != 5:
            return action_idx, coords, coord_idx
        if self.rng.random() >= OBJCLICK_REDRAW:
            self.stats["objclick_kept"] += 1
            return action_idx, coords, coord_idx
        size, n_shapes = self.shapes(raw)
        w = np.asarray(all_probs[5:], dtype=np.float64) / size
        total = w.sum()
        if not np.isfinite(w).all() or not total > 0:           # nothing to draw from: Goose's cell stands
            self.stats["objclick_kept"] += 1
            self.stats["objclick_no_weights"] += 1
            return action_idx, coords, coord_idx
        c = int(self.rng.choice(GRID * GRID, p=w / total))
        self.stats["objclick_redrawn"] += 1
        self.stats["objclick_moved"] += int(c != coord_idx)
        self.stats["objclick_shapes"] += n_shapes
        return 5, (c // GRID, c % GRID), c

    def summary(self):
        """Clicks redrawn and kept, and the mean number of shapes on a screen it redrew on."""
        n = self.stats["objclick_redrawn"]
        return dict(self.stats, objclick_mean_shapes=round(self.stats["objclick_shapes"] / n, 2) if n else 0.0)


# --------------------------------------------------------------------------------
# The upgrades object the agent talks to
# --------------------------------------------------------------------------------
class Upgrades:
    def __init__(self, names, rng=None, seed=0):
        names = {n.strip() for n in names if n.strip()}
        gone = names & REMOVED
        if gone:
            raise SystemExit(f"EVAL_UPGRADES: {sorted(gone)} lost the upgrade screen and were removed "
                             f"on 2026-10-06; run them from git tag upgrade-screen-round1")
        unknown = names - KNOWN
        if unknown:
            raise SystemExit(f"EVAL_UPGRADES: unknown {sorted(unknown)}; known {sorted(KNOWN)}")
        if {"map_gated", "map_sched"} <= names:
            raise SystemExit("EVAL_UPGRADES: map_sched takes map_gated's place; ask for one of them")
        self.names = names
        self.rng = rng if rng is not None else np.random
        self.bars = BarDetector() if "bars" in names else None
        # carry.py: its own generator, seeded from the agent's seed, so Goose's stream is untouched
        self.carry = Carry("carry_noops" in names, "carry_hints" in names, seed) if names & CARRY_OPTIONS else None
        self.objclick = ObjClick(seed) if "objclick" in names else None     # its own generator too
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
    def from_env(cls, seed=0):
        return cls(os.getenv("EVAL_UPGRADES", "").split(","), seed=seed)

    def check(self, map_on):
        if self.names & MAP_OPTIONS and not map_on:
            raise SystemExit(f"EVAL_UPGRADES {sorted(self.names & MAP_OPTIONS)} need EVAL_RETURN_MAP=1")
        if "map_sched" in self.names and os.getenv("EVAL_GB", "").strip().lower() in ("1", "true", "yes", "on"):
            raise SystemExit("EVAL_UPGRADES map_sched is not wired into Glassbox's map (GbMap): "
                             "run it without EVAL_GB")

    def config(self):
        out = {"upgrades": ",".join(sorted(self.names))}
        if self.carry is not None:
            out["carry_settings"] = self.carry.settings()
        if "map_sched" in self.names:
            out["map_sched_settings"] = {
                "schedules": {name: {"stall": st, "walk_share": share, "untried_first": untried}
                              for name, (st, share, untried) in SCHEDULES.items()},
                "start": SCHED_START, "window": SCHED_WINDOW, "min_tries": SCHED_MIN_TRIES,
                "stretch": SCHED_STRETCH}
        if self.objclick is not None:
            out["objclick_settings"] = {"redraw": OBJCLICK_REDRAW}
        return out

    def wrap_map(self, m):
        """Replace the return map with the upgraded subclass (same settings)."""
        if not (self.names & MAP_OPTIONS) and self.bars is None:
            return m
        return UpgradedReturnMap(stall=m.stall, click_tries=m.click_tries, max_route=m.max_route,
                                 return_after_game_over=m.return_after_game_over,
                                 max_nodes=m.max_nodes, rng=m.rng,
                                 gated="map_gated" in self.names, bars=self.bars,
                                 sched="map_sched" in self.names)

    # ---- per step -----------------------------------------------------------------
    def live_mask(self, shared_mask):
        """The decoration mask: the shared detector's, plus the `bars` upgrade's sticky cells."""
        return shared_mask if self.bars is None else (shared_mask | self.bars.sticky)

    def key(self, raw, shared_mask):
        m = self.live_mask(shared_mask)
        g = np.array(raw, dtype=np.uint8, copy=True)
        g[m] = 0
        return xxhash.xxh64(g.tobytes()).intdigest()

    def observe(self, prev_raw, cur_raw, shared_mask, prev_action):
        """Every decision frame. prev_raw is None right after a reset / level
        change / skipped frame (no transition to learn from)."""
        if self.carry is not None:      # before the bars take this move: the order S5 saw moves in
            self.carry.step(prev_raw, prev_action, cur_raw, self.carry.live_mask(shared_mask, self.bars))
        if self.bars is not None and prev_raw is not None:
            before = self.bars.growths
            self.bars.update(prev_raw, cur_raw)
            self.stats["bar_growths"] += self.bars.growths - before
        if self.objclick is not None:   # the mask the key below uses, for this decision's click
            self.objclick.mask = self.live_mask(shared_mask)
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
        """Goose's sampled move, after `objclick`, `deadclick` and then the carry upgrades
        have had their say (each returns the move unchanged when it is off)."""
        move = (action_idx, coords, coord_idx)
        if self.objclick is not None:
            move = self.objclick.override(*move, all_probs, cur_raw)
        move = self._dead_click(*move, all_probs, cur_raw)
        if self.carry is not None:
            move = self.carry.override(*move, all_probs, cur_raw)
        return move

    def _dead_click(self, action_idx, coords, coord_idx, all_probs, cur_raw):
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
        if self.carry is not None:
            self.carry.on_level()
        self.visits = {}
        self.seen_attempt = set()
        self.dead = {}
        self.blocked = set()
        self.prev_key = None

    def summary(self):
        out = dict(self.stats, bar_cells=int(self.bars.sticky.sum()) if self.bars is not None else 0)
        if self.carry is not None:
            out["carry"] = self.carry.summary()
        if self.objclick is not None:
            out.update(self.objclick.summary())
        return out

    def dump(self, path, return_map=None):
        out = dict(self.summary(), **self.config())
        try:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
        except OSError:
            pass
