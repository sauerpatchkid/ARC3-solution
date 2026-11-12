"""carry.py — what one level teaches Goose about the next, with no LLM.

THE PROBLEM IT FIXES
Goose starts every level from nothing (EVAL_RESET_ON_LEVEL): a new network, an
empty buffer. Two things it learned on the level before are thrown away:

  NO-OPS  which colours do nothing when clicked. On a new level Goose spends its
          first thousands of moves finding that out again.
  GOAL    what the level was about. Goose has no idea what to aim for.

WHAT IT DOES (two switches in EVAL_UPGRADES, each usable alone)

  carry_noops  A colour clicked at least CARRY_MIN_CLICKS times on one level, whose
               clicks never changed a live cell, is a no-op on every later level:
               a click on it keeps only CARRY_ALPHA of its probability (never zero,
               so a contradiction can still show up). "Never changed a live cell"
               is judged at the level's end, under the decoration mask of that
               moment with its bars run to the screen's edges, so a click that only
               moved a decoration early in the level does not count. The first
               click on a carried colour that does change a live cell drops it.
  carry_hints  At each level-up, the CARRY_HINTS colours that vanished or shrank
               most between the level's first board and the board before the
               winning move (never the background) become goal hints. On the next
               level a click on a hint colour gets x (1 + CARRY_GAMMA), only on
               screens where such a click is possible. A hint is dropped after
               CARRY_HINT_DROP moves in which no cell of that colour changed.

How a weight reaches Goose. Goose samples a move from its own distribution, as
always. The move is accepted with probability w(move) / w_max and otherwise drawn
again from the same distribution with THIS module's own generator. That is
rejection sampling: the move taken has probability p(move) w(move) / sum(p w), and
Goose's random stream is untouched.

WHERE IT COMES FROM
This is Glassbox's S5 v2 guide run with no rule book (custom_agents/gb/s5_act.py:
Guide and Online with `colour_judge = "level_end"` and `mask_bars = true`; the arm
called GB-free). The numbers above are that guide's settings. It is a PORT, not an
import, so that this Goose upgrade does not depend on Glassbox. With both switches
on and the same seed it makes the same moves as GB-free, move for move:
tests/test_carry.py checks the two against each other on made-up levels, and
docs/1-goose-upgrades/carry-inside-goose.md records the check on a real game.

Evidence so far (docs/3-glassbox/time-to-level.md, exploratory): with this guide,
levels after the first were finished in fewer actions 36 times against 17 in G4,
where two unguided arms split 24 against 27. The level count stayed inside the
noise. Its own test is in docs/1-goose-upgrades/carry-inside-goose.md.

HOW TO REMOVE IT
Leave `carry_noops` and `carry_hints` out of EVAL_UPGRADES (the default) and
nothing here runs. To delete it: this file, tests/test_carry.py, and the lines in
custom_agents/upgrades.py that mention `carry`.

Pure numpy; no torch.
"""
import collections

import numpy as np

from gridtools import GRID, complete_bars

CLICKS = 5                  # unified action index of the first click (5 + 64*y + x)
CARRY_ALPHA = 0.1           # a click on a carried no-op colour keeps this share of its probability
CARRY_GAMMA = 1.0           # a click on a goal-hint colour gets x (1 + gamma)
CARRY_MIN_CLICKS = 20       # clicks on a colour, on one level, before it can be carried
CARRY_HINT_DROP = 2000      # moves without a change on a hint colour before the hint is dropped
CARRY_HINTS = 2             # goal-hint colours kept per level
CARRY_MAX_DRAWS = 100       # rejection sampling: the move drawn last is taken
CARRY_SEED = 7919           # added to the agent's seed for this module's own generator (as S5 does)


def clicked_colour(board, action):
    y, x = divmod(int(action) - CLICKS, GRID)
    return int(board[y, x])


def goal_colours(first, before_win, mask, n=CARRY_HINTS):
    """The `n` colours that vanished or shrank most from a level's first board to the
    board before its winning move (cells outside the decoration mask), never the
    first board's background (its most common live colour: it "shrinks" whenever
    anything appears)."""
    live = ~np.asarray(mask, bool)
    a = np.bincount(np.asarray(first)[live].ravel(), minlength=16)
    b = np.bincount(np.asarray(before_win)[live].ravel(), minlength=16)
    drop = a - b
    drop[int(a.argmax())] = 0
    order = sorted((c for c in range(16) if drop[c] > 0), key=lambda c: (b[c] > 0, -drop[c], c))
    return order[:n]


class Carry:
    """Carried colour no-ops and goal hints for one game, level by level (module doc).
    custom_agents/upgrades.py calls `on_level` when the score changes, `step` on every
    decision frame and `override` on every move Goose samples."""

    def __init__(self, noops=True, hints=True, seed=0):
        self.noops, self.use_hints = bool(noops), bool(hints)
        self.rng = np.random.default_rng(seed + CARRY_SEED)
        self.level = 0                          # 1-based; 0 before the first level starts
        self.pending = False                    # the score just changed: the next screen is a level's first
        self.colour_noops = {}                  # colour -> level it was carried from
        self.hints = {}                         # colour -> moves since a cell of it changed
        self.first = self.mask = None           # this level's first board, and the mask at its start
        self.last = None                        # the last board seen (at a level-up: the one before the win)
        self.clicks = collections.Counter()     # colour -> clicks this level
        self.click_cells = {}                   # colour -> cells its clicks changed this level (unmasked)
        self.moves_on_level = 0
        self.stats = collections.Counter()
        self.events = []
        self._mask_key = self._mask_done = None

    # -- the mask ---------------------------------------------------------------------------------
    def live_mask(self, shared_mask, bars=None):
        """The agent's decoration mask (the shared detector's, plus the `bars` upgrade's
        sticky cells) with each bar run to the screen's edges. Worked out again only
        when the mask has changed (it changes at the detectors' refreshes)."""
        m = np.zeros((GRID, GRID), bool) if shared_mask is None else np.asarray(shared_mask, bool)
        if bars is not None:
            m = m | bars.sticky
        key = np.packbits(m).tobytes()
        if key != self._mask_key:
            self._mask_key, self._mask_done = key, complete_bars(m)
        return self._mask_done

    # -- what the agent tells it ------------------------------------------------------------------
    def on_level(self):
        self.pending = True

    def step(self, prev_raw, prev_action, cur_raw, mask):
        """Every decision frame. `prev_raw` is None right after a reset or a level change."""
        if self.pending:                        # the mask now is also the finished level's end mask
            self._new_level(cur_raw, mask)
            self.pending = False
        elif prev_raw is not None and prev_action is not None:
            self._observe(prev_raw, int(prev_action), cur_raw, bool(((prev_raw != cur_raw) & ~mask).any()))
        self.last = np.array(cur_raw, np.uint8)

    def _event(self, kind, **kw):
        self.events.append(dict(kind=kind, level=self.level, move=self.moves_on_level, **kw))

    def _new_level(self, first, mask):
        if self.level:                          # a level has just been finished: what does it teach?
            end = complete_bars(np.asarray(mask, bool))
            if self.noops:
                for c in sorted(self.clicks):
                    cells = self.click_cells.get(c)
                    if self.clicks[c] >= CARRY_MIN_CLICKS and c not in self.colour_noops \
                            and (cells is None or not (cells & ~end).any()):
                        self.colour_noops[c] = self.level
                        self._event("colour no-op carried", colour=c, clicks=self.clicks[c])
            if self.use_hints and self.last is not None and self.first is not None:
                self.hints = {c: 0 for c in goal_colours(self.first, self.last, self.mask)}
                if self.hints:
                    self._event("goal hints", colours=sorted(self.hints))
        self.level += 1
        self.first, self.mask = np.array(first, np.uint8), np.asarray(mask, bool)
        self.clicks, self.click_cells, self.moves_on_level = collections.Counter(), {}, 0

    def _observe(self, before, action, after, changed):
        """One move made on the current level (unified action index)."""
        self.moves_on_level += 1
        if changed and self.covers(before, action):         # a contradiction: drop the colour
            c = clicked_colour(before, action)
            del self.colour_noops[c]
            self._event("colour no-op dropped", colour=c)
            self.stats["contradictions"] += 1
        if action >= CLICKS:
            c = clicked_colour(before, action)
            self.clicks[c] += 1
            diff = before != after
            self.click_cells[c] = self.click_cells[c] | diff if c in self.click_cells else diff
        if self.hints:
            diff = (before != after) & ~self.mask
            for c in list(self.hints):
                self.hints[c] = 0 if (diff & ((before == c) | (after == c))).any() else self.hints[c] + 1
                if self.hints[c] >= CARRY_HINT_DROP:
                    del self.hints[c]
                    self._event("goal hint dropped", colour=c)

    # -- what it tells the agent ------------------------------------------------------------------
    def covers(self, board, action):
        """Is this move a click on a carried no-op colour?"""
        return self.noops and action >= CLICKS and clicked_colour(board, action) in self.colour_noops

    def weight(self, board, action):
        w = CARRY_ALPHA if self.covers(board, action) else 1.0
        if self.hints and action >= CLICKS and clicked_colour(board, action) in self.hints:
            w *= 1.0 + CARRY_GAMMA
        return w

    def override(self, action_idx, coords, coord_idx, all_probs, board):
        """Goose's sampled move, accepted with probability w / w_max, else drawn again
        from Goose's distribution with this module's generator (module doc). Returns the
        triple Goose's sampler returns: (action_idx, (y, x) or None, coord_idx or None)."""
        a = action_idx if action_idx < CLICKS else CLICKS + coord_idx
        w_max = 1.0
        if self.hints:                              # a boost only where a hint-coloured click is possible
            clicks = np.asarray(all_probs[CLICKS:]).reshape(GRID, GRID)
            if (clicks[np.isin(board, list(self.hints))] > 0).any():
                w_max = 1.0 + CARRY_GAMMA
        p, first = None, a
        for _ in range(CARRY_MAX_DRAWS):
            w = self.weight(board, a)
            self.stats["draws"] += 1
            if w >= w_max or self.rng.random() < w / w_max:
                break
            if p is None:
                p = np.asarray(all_probs, np.float64).copy()
                p[CLICKS:] /= GRID * GRID
                if not np.isfinite(p).all() or p.sum() <= 0:
                    break
                p /= p.sum()
            a = int(self.rng.choice(len(p), p=p))
        self.stats["moves"] += 1
        self.stats["covered_first"] += int(bool(self.covers(board, first)))
        self.stats["changed_move"] += int(a != first)
        if a < CLICKS:
            return a, None, None
        c = a - CLICKS
        return CLICKS, (c // GRID, c % GRID), c

    # -- reporting --------------------------------------------------------------------------------
    def settings(self):
        return {"noops": self.noops, "hints": self.use_hints, "alpha": CARRY_ALPHA, "gamma": CARRY_GAMMA,
                "min_clicks": CARRY_MIN_CLICKS, "hint_drop_moves": CARRY_HINT_DROP, "n_hints": CARRY_HINTS}

    def summary(self):
        kinds = collections.Counter(e["kind"] for e in self.events)
        return dict({k: int(v) for k, v in self.stats.items()},
                    colours_carried=int(kinds["colour no-op carried"]), colours_dropped=int(kinds["colour no-op dropped"]),
                    hints_set=int(kinds["goal hints"]), hints_dropped=int(kinds["goal hint dropped"]),
                    colour_noops=sorted(self.colour_noops), hints=sorted(self.hints), events=self.events)
