"""evidence.py - one level's recorded moves, as the evidence rules are checked on
(docs/plans/llm-rulebook.md section 3.1).

A KEY is a distinct (screen, action): the screen with decoration cells blanked,
plus the unified action index. Every recorded move of a level falls on one key.
For each key the index keeps how many moves landed on it and every distinct
OUTCOME (next screen, decorations blanked) with its count. Nothing is thrown
away before outcomes are compared: a key with more than one outcome is a
CONFLICT (a masked resource, a mask error, history the screen doesn't show, or
truly hidden state) and is flagged, never silently deduplicated.

The check set is every key (capped by a uniform sample when a level has more
than `max_keys`), each with its MAJORITY outcome. Moves inside a frozen stretch
(more than 500 moves in a row that change nothing at all) stay in the check set
but are never shown in a prompt.

The level-completing move is left out: its next screen is the next level's
first screen, which no rule about mechanics can predict.

MOVES (Rulebook v2, docs/plans/rulebook-v2-prereg.md section 4.4). The index
above merges moves made at different times, so beside it `moves` keeps one entry
per recorded move, in order: `key` (which key it fell on, -1 if the cap dropped
that key), `out` (the hash of its OWN next screen, decorations blanked) and,
when the recording has action numbers, `num` (the action number), `attempt`
(which attempt of this level, from 0) and `t` (recorded moves since that attempt
started). An attempt ends at a game over: the move flagged `game_over`, or in
recordings made before 2026-10-06 (which have no flag and did not record that
move) the action number stepping by more than 1. Evidence files saved before
this was added load with `moves` = None.

MASK. Per level, from that level's own moves: the component-level ticker scan
(copied from legacy/llm_track/tickers.py), limited to the screen's outer 4-cell
band as mb_gated_att's bar detector is, with the gaps inside each ticker row
filled. Filling is an inference - Stage A found rarely-changing bar cells that
alone kept a correct rule at 94% - so the filled cells are counted in `info`.

Conventions: frames are 64x64 uint8 colour indices; action_idx 0-4 = ACTION1-5,
5 + 64*row + col = a click; `level` is the score when the move was made.
"""
import collections
import glob
import os

import numpy as np
import xxhash

from gridtools import BAR_BORDER, border_band, connected_components, tick_cells  # noqa: F401
# (connected_components is re-exported: wm/rules.py and the tests import it from here)

GRID = 64
PALETTE_NAMES = ["white", "light gray", "gray", "dark gray", "darker gray", "black",
                 "pink", "light pink", "red", "blue", "light blue", "yellow",
                 "orange", "dark red", "green", "purple"]
FROZEN = 500                 # longer no-change-at-all stretches are a frozen screen
MIN_GROUP = 30               # action groups with fewer recorded moves get no prompt
MAX_KEYS = 40000             # check-set cap per level (uniform sample above it)
MASK_MOVES = 20000           # moves of the level the mask is estimated from
# Ticker scan thresholds: identical to legacy/llm_track/tickers.py.
DECOR_THRESHOLD, MAX_TICK_CELLS, MIN_TICK_FRAC, TICK_COVERAGE, MAX_MASK_CELLS = 0.95, 2, 0.30, 0.95, 256


# --------------------------------------------------------------------------------
# What a rule is given
# --------------------------------------------------------------------------------
class Act:
    """The `act` argument: what the player did."""
    __slots__ = ("action", "click")

    def __init__(self, action_idx):
        a = int(action_idx)
        if a < 5:
            self.action, self.click = a + 1, None
        else:
            self.action, self.click = 6, divmod(a - 5, GRID)

    def __repr__(self):
        return f"Act(action={self.action}, click={self.click})"


class Region:
    """A connected same-colour area of a board (8-connected)."""
    __slots__ = ("colour", "size", "y0", "x0", "y1", "x1", "cells", "mask")

    def __init__(self, colour, cells):
        self.colour = int(colour)
        self.cells = cells                               # (n, 2) int16: (row, col)
        self.mask = np.zeros((GRID, GRID), dtype=bool)
        self.mask[cells[:, 0], cells[:, 1]] = True
        self.size = int(len(cells))
        self.y0, self.x0 = (int(v) for v in cells.min(axis=0))
        self.y1, self.x1 = (int(v) for v in cells.max(axis=0))

    def __repr__(self):
        return (f"Region(colour={self.colour}, size={self.size}, "
                f"rows {self.y0}-{self.y1}, cols {self.x0}-{self.x1})")


class Api:
    """The `api` argument: what a rule may know about the level.

    Rulebook v2 (docs/plans/rulebook-v2.md section 4.1) adds `level` (1 for the
    first level, as the prompts count them), `colours` (the colours on the
    level's first board) and, for a rule graded move by move, `t` (recorded
    moves since the attempt started) and `attempt`; those two are None when a
    rule is checked on cases, which merge moves made at different times."""
    __slots__ = ("first", "background", "ticker", "layout", "level", "colours", "t", "attempt")

    def __init__(self, first, ticker, level=None):
        self.first = _readonly(first)
        self.ticker = _readonly(ticker)
        self.level = None if level is None else int(level) + 1
        self.colours = tuple(int(c) for c in np.unique(first))
        self.t = self.attempt = None
        self.background = int(np.bincount(np.asarray(first).ravel(), minlength=16).argmax())
        lay = []
        for c in np.unique(first):
            if int(c) == self.background:
                continue
            m = (first == c) & ~ticker
            if m.any():
                lay += [Region(c, cells) for cells in connected_components(m)]
        lay.sort(key=lambda r: -r.size)
        for r in lay:
            r.cells.flags.writeable = False
            r.mask.flags.writeable = False
        self.layout = lay


def group_of(action_idx, before):
    """Action group: each button, and clicks by the colour clicked."""
    a = int(action_idx)
    if a < 5:
        return f"ACTION{a + 1}"
    y, x = divmod(a - 5, GRID)
    c = int(before[y, x])
    return f"click on colour {c} ({PALETTE_NAMES[c]})"


def _readonly(a):
    a = np.array(a)
    a.flags.writeable = False
    return a


# --------------------------------------------------------------------------------
# Decoration mask (the scan is legacy/llm_track/tickers.py's; helpers in gridtools.py)
# --------------------------------------------------------------------------------
def ticker_mask(frames, next_frames):
    """(mask, info): cells that change by themselves. FIXED = changes in >= 95%
    of moves; ROTATING = the compact cell set covering 95% of the changes made
    by components of at most 2 cells, when at least 30% of moves contain one,
    kept only within BAR_BORDER cells of a screen edge. Then the gaps inside
    each masked row are filled.

    The edge guard is upgrades.BarDetector's, measured on the corpora: without
    it a small object sliding across the playfield reads as a ticker (it
    flagged mid-screen rows on sk48, s5i5 and tr87), and every real bar found
    sat on rows 0-3 or 60-63 or column 0."""
    diff = frames != next_frames
    n = len(diff)
    if n == 0:
        return np.zeros((GRID, GRID), bool), {"n": 0}
    tick_count = np.zeros((GRID, GRID), np.int64)
    n_tick = 0
    for d in diff:                      # gridtools.tick_cells: changed blobs of <= MAX_TICK_CELLS
        if not d.any():
            continue
        t = tick_cells(d)
        if t.any():
            tick_count += t
            n_tick += 1
    fixed = diff.mean(axis=0) >= DECOR_THRESHOLD
    rot = np.zeros(GRID * GRID, bool)
    if n_tick / n >= MIN_TICK_FRAC and tick_count.sum() > 0:
        counts = tick_count.ravel().astype(float)
        order = np.argsort(-counts)
        k = int(np.searchsorted(np.cumsum(counts[order]) / counts.sum(), TICK_COVERAGE)) + 1
        if k <= MAX_MASK_CELLS:
            rot[order[:k]] = True
    raw = fixed | (rot.reshape(GRID, GRID) & border_band())
    mask = raw.copy()
    for y in np.flatnonzero(raw.any(axis=1)):
        xs = np.flatnonzero(raw[y])
        mask[y, xs.min():xs.max() + 1] = True
    return mask, {"n": n, "tick_frac": round(n_tick / n, 4), "detected_cells": int(raw.sum()),
                  "filled_cells": int(mask.sum() - raw.sum()), "mask_cells": int(mask.sum())}


def frozen_moves(changed):
    """True for moves inside a stretch of more than FROZEN consecutive moves that
    changed nothing on screen (decorations included)."""
    x = np.diff(np.r_[True, np.asarray(changed, bool), True].astype(np.int8))
    out = np.zeros(len(changed), bool)
    for s, e in zip(np.flatnonzero(x == -1), np.flatnonzero(x == 1)):
        if e - s > FROZEN:
            out[s:e] = True
    return out


# --------------------------------------------------------------------------------
# Reading a level out of a recorded run
# --------------------------------------------------------------------------------
def screen_hash(board, live):
    """The hash of a board with decoration cells blanked: what a move's `out`
    and a checked rule's `phash` are, so the two can be compared."""
    return xxhash.xxh64(np.where(live, board, 255).astype(np.uint8).tobytes()).intdigest()


def read_moves(corpus, level):
    """All moves a run made at `level`, in order, without the level-completing
    move: (frames, next_frames, actions, order). `order` is {num, attempt, t}
    per move (see MOVES in the module doc), or None if the recording has no
    action numbers. Empty arrays if the level was never played."""
    fs, ns, acts, levs, nums, overs = [], [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(corpus, "shard_*.npz"))):
        with np.load(p) as z:
            lv = z["levels"]
            sel = (lv >= level) & (lv <= level + 1)
            if sel.any():
                fs.append(z["frames"][sel])
                ns.append(z["next_frames"][sel])
                acts.append(z["actions"][sel])
                levs.append(lv[sel])
                if "action_nums" in z.files:
                    nums.append(z["action_nums"][sel])
                if "game_over" in z.files:
                    overs.append(z["game_over"][sel])
    if not fs:
        e = np.zeros((0, GRID, GRID), np.uint8)
        return e, e, np.zeros(0, np.int32), None
    f, n, a, lv = (np.concatenate(x) for x in (fs, ns, acts, levs))
    here = lv == level
    # the last move at this level before the next level's first move completed it
    idx = np.flatnonzero(here)
    if len(idx) and (lv == level + 1).any() and idx[-1] + 1 < len(lv) and lv[idx[-1] + 1] == level + 1:
        here[idx[-1]] = False
    order = None
    if len(nums) == len(fs):
        num = np.concatenate(nums)[here].astype(np.int64)
        if len(overs) == len(fs):                     # recorded since 2026-10-06: the flagged move ends its attempt
            new = np.r_[False, np.concatenate(overs)[here][:-1] > 0]
        else:                                         # before: the move itself is missing from the recording
            new = np.r_[False, np.diff(num) > 1]
        starts = np.flatnonzero(np.r_[True, new[1:]]) if len(num) else np.zeros(0, np.int64)
        attempt = np.cumsum(new).astype(np.int32)
        order = {"num": num, "attempt": attempt,
                 "t": (np.arange(len(num)) - starts[attempt]).astype(np.int32)}
    return f[here], n[here], a[here].astype(np.int32), order


def read_level(corpus, level):
    """read_moves without the order: (frames, next_frames, actions)."""
    return read_moves(corpus, level)[:3]


class LevelEvidence:
    """The check set for one level. Arrays are per KEY (see the module doc)."""

    def __init__(self, game, level, first, mask, info, before, after, actions, count, agree,
                 n_variants, frozen, moves=None):
        self.game, self.level, self.info = game, int(level), info
        self.moves = moves                                   # per recorded move (module doc), or None
        self.first, self.mask = _readonly(first), _readonly(mask)
        self.live = ~self.mask
        self.before, self.after, self.actions = before, after, actions
        self.count, self.agree, self.n_variants, self.frozen = count, agree, n_variants, frozen
        self.changed = ((before != after) & self.live).reshape(len(actions), -1).any(axis=1)
        self.group = [group_of(a, b) for a, b in zip(actions, before)]
        moves = collections.Counter()
        for g, c in zip(self.group, count):
            moves[g] += int(c)
        self.group_moves = dict(moves)
        self.groups = sorted((g for g, c in moves.items() if c >= MIN_GROUP), key=lambda g: (-moves[g], g))
        self._api = None

    # -- building --------------------------------------------------------------
    @classmethod
    def from_moves(cls, game, level, frames, next_frames, actions, max_keys=MAX_KEYS, seed=0, mask=None,
                   order=None):
        """Index a level's moves. `mask`: reuse a mask instead of estimating one
        (tests; a later re-estimate would bump the mask version). `order`:
        read_moves' per-move {num, attempt, t}, kept beside the index."""
        if mask is None:
            mask, info = ticker_mask(frames[:MASK_MOVES], next_frames[:MASK_MOVES])
        else:
            info = {"n": len(frames), "mask_cells": int(mask.sum()), "given": True}
        live = ~mask
        raw_changed = (frames != next_frames).reshape(len(frames), -1).any(axis=1)
        frozen = frozen_moves(raw_changed)
        keys = {}                          # key -> [count, {outcome hash: [count, move index]}, all frozen, number]
        move_key, move_out = np.zeros(len(actions), np.int64), np.zeros(len(actions), np.uint64)
        for i in range(len(actions)):
            b = np.where(live, frames[i], 255).tobytes()
            k = (xxhash.xxh64(b).intdigest(), int(actions[i]))
            o = xxhash.xxh64(np.where(live, next_frames[i], 255).tobytes()).intdigest()
            rec = keys.get(k)
            if rec is None:
                rec = keys[k] = [0, {}, True, len(keys)]
            move_key[i], move_out[i] = rec[3], o
            rec[0] += 1
            v = rec[1].get(o)
            if v is None:
                rec[1][o] = [1, i]
            else:
                v[0] += 1
            rec[2] = rec[2] and bool(frozen[i])
        recs = list(keys.values())
        n_keys = len(recs)
        if n_keys > max_keys:
            pick = np.sort(np.random.default_rng(seed).choice(n_keys, max_keys, replace=False))
            recs = [recs[j] for j in pick]
            kept = np.full(n_keys, -1, np.int64)
            kept[pick] = np.arange(max_keys)
            move_key = kept[move_key]
        major = [max(r[1].values(), key=lambda v: (v[0], -v[1])) for r in recs]
        idx = np.array([m[1] for m in major], dtype=np.int64)
        info = dict(info, moves=int(len(actions)), keys=n_keys, keys_kept=len(recs),
                    frozen_moves=int(frozen.sum()),
                    conflict_keys=sum(1 for r in recs if len(r[1]) > 1),
                    conflict_moves=sum(r[0] - m[0] for r, m in zip(recs, major)))
        first = frames[0] if len(frames) else np.zeros((GRID, GRID), np.uint8)
        return cls(game, level, first, mask, info,
                   before=frames[idx].astype(np.uint8), after=next_frames[idx].astype(np.uint8),
                   actions=actions[idx].astype(np.int32),
                   count=np.array([r[0] for r in recs], np.int64),
                   agree=np.array([m[0] for m in major], np.int64),
                   n_variants=np.array([len(r[1]) for r in recs], np.int32),
                   frozen=np.array([r[2] for r in recs], bool),
                   moves=dict(order or {}, key=move_key.astype(np.int32), out=move_out))

    @classmethod
    def from_corpus(cls, game, corpus, level, **kw):
        f, n, a, order = read_moves(corpus, level)
        return cls.from_moves(game, level, f, n, a, order=order, **kw) if len(a) else None

    @classmethod
    def split(cls, game, corpus, level, fit_moves=300, known_mask=None, **kw):
        """(fit, score) for a transfer level (docs/plans/rulebook-v2.md section
        3.3): the level's first `fit_moves` moves are what re-binding and
        repair may look at, the rest are what scores are computed on, so no
        arm is scored on moves it was fitted to.

        Masks. `score` uses the level's own mask, estimated as for any level
        (so it is the same mask v1 scored this level with). `fit` uses only
        what is known by then: the ticker scan over the fit moves themselves,
        joined with `known_mask` (the training level's mask, which a live agent
        carries over). `fit` is None if the level has no moves; `score` is None
        if it has `fit_moves` or fewer."""
        f, n, a, order = read_moves(corpus, level)
        if not len(a):
            return None, None
        level_mask, _ = ticker_mask(f[:MASK_MOVES], n[:MASK_MOVES])
        fit_mask, _ = ticker_mask(f[:fit_moves], n[:fit_moves])
        if known_mask is not None:
            fit_mask = fit_mask | np.asarray(known_mask, bool)
        part = lambda s: {k: v[s] for k, v in order.items()} if order else None   # attempts are the LEVEL's
        fit = cls.from_moves(game, level, f[:fit_moves], n[:fit_moves], a[:fit_moves], mask=fit_mask,
                             order=part(slice(None, fit_moves)), **kw)
        score = (cls.from_moves(game, level, f[fit_moves:], n[fit_moves:], a[fit_moves:], mask=level_mask,
                                order=part(slice(fit_moves, None)), **kw)
                 if len(a) > fit_moves else None)
        if score is not None:
            score.first = _readonly(f[0])     # the level's first board, not the split's
            score._api = None
        return fit, score

    def save(self, path):
        np.savez_compressed(path, game=self.game, level=self.level, first=self.first, mask=self.mask,
                            info=np.array(repr(self.info)), before=self.before, after=self.after,
                            actions=self.actions, count=self.count, agree=self.agree,
                            n_variants=self.n_variants, frozen=self.frozen,
                            **{"move_" + k: v for k, v in (self.moves or {}).items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(str(z["game"]), int(z["level"]), z["first"], z["mask"], {"loaded": True},
                   z["before"], z["after"], z["actions"], z["count"], z["agree"], z["n_variants"],
                   z["frozen"], moves={k[5:]: z[k] for k in z.files if k.startswith("move_")} or None)

    # -- what rules and the checker use ----------------------------------------
    @property
    def api(self):
        if self._api is None:
            self._api = Api(self.first, self.mask, self.level)
        return self._api

    def inputs(self, k):
        """(board, act, api), read-only, as a live agent would build them."""
        return _readonly(self.before[k]), Act(self.actions[k]), self.api

    def exact(self, k, pred):
        pred = np.asarray(pred)
        return pred.shape == (GRID, GRID) and bool(np.array_equal(pred[self.live], self.after[k][self.live]))

    def keys_of(self, group):
        return np.array([i for i, g in enumerate(self.group) if g == group], dtype=np.int64)

    def summary(self):
        ch = self.changed
        return dict(self.info, level=self.level + 1, groups=len(self.groups),
                    changing_keys=int(ch.sum()), unchanged_keys=int((~ch).sum()))
