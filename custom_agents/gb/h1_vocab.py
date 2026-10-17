"""h1_vocab.py - H1 VOCAB: the object and relation code that S1, S3's guards and the rules share.

The problem it fixes: S1 tells the LLM what is next to the object a move is
about (touching, a ray, the screen edge), S3's guard fitter learns from those
same facts when a rule fails, and the guard it writes must work them out again
inside the rule sandbox, from the board alone. Three copies of that code would
drift apart, and a guard fitted on one copy would be judged by another. This
file is the one copy. (Plan section 5.7.) It also holds the grid operations rules
may call (H1's primitives, below), so the LLM writes short rules from tested
parts instead of raw numpy with the same boundary bugs every time.

  Obj, objects        4-connected single-colour regions of live cells that are
                      not the background (S1 step 1)
  subject             the object a move is about: the clicked object, or the
                      largest object of the mover colour (S1 step 4)
  sits_on, relations, inside    what is around it (S1 step 4)
  feature_names, feature_set, feature_row    those facts as named yes/no bits
                      (S1 step 5)
  features            the same bits inside the rule sandbox, from what a rule
                      is given (board, act, api) plus the mover colour: what a
                      fitted guard calls
  PRIMITIVES          objects, find_object, object_at, ray, is_blocked,
                      translate, slide, recolour, swap, unchanged: the plan's
                      cut-line ten (section 5.7), each tested on synthetic boards
                      (tests/test_gb_vocab.py), described to the LLM by CARD.
                      Version 1 is registered as glassbox_vocab_v1 before G1;
                      a change after that is version 2.

SANDBOX is what gb/core/contract.py hands every rule beside numpy and the slot
types. In the sandbox the subject is found by flood fill over row runs, never by
labelling the whole board, so a guard costs well under the 5 ms a rule may take
per move. `subject_inside` alone needs every object on the board, so it is only
worked out when a guard asks for it. S1 finds its subject with the same
`subject()` and checks that it is one of its own labelled objects, so the two
cannot drift apart without S1 stopping.
"""
import numpy as np

from gb.core.evidence import PALETTE_NAMES
from gridtools import GRID, label_components

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
LARGE_CELLS = 1024           # an object over this many cells is `large` (walls, floors): S1's default
RAY_MAX = 64                 # how far a relation looks: S1's default
SIZE_BITS = (1, 4, 16, 64)
RAY_BITS = (2, 4, 8)


# ---- objects -------------------------------------------------------------------------------------
class Obj:
    """One single-colour region: its colour, its cells (flat indices, sorted), box
    (top, bottom, left, right; inclusive), shape and, for rules, `mask` (a
    read-only 64x64 bool array, True on its cells)."""
    __slots__ = ("colour", "cells", "size", "box", "shape", "large", "_m")

    def __init__(self, colour, cells, large_cells=LARGE_CELLS):
        self.colour, self.cells = int(colour), cells
        self.size = len(cells)
        ys, xs = cells // GRID, cells % GRID
        self.box = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        self.shape = (self.box[1] - self.box[0], self.box[3] - self.box[2],
                      ((ys - self.box[0]) * GRID + (xs - self.box[2])).tobytes())    # cells relative to the box
        self.large = self.size > large_cells
        self._m = None

    @property
    def mask(self):
        if self._m is None:
            m = np.zeros(GRID * GRID, bool)
            m[self.cells] = True
            self._m = m.reshape(GRID, GRID)
            self._m.flags.writeable = False
        return self._m

    def describe(self):
        r0, r1, c0, c1 = self.box
        rows = f"row {r0}" if r0 == r1 else f"rows {r0}-{r1}"
        cols = f"col {c0}" if c0 == c1 else f"cols {c0}-{c1}"
        return (f"colour {self.colour} {PALETTE_NAMES[self.colour]}, {r1 - r0 + 1}x{c1 - c0 + 1}"
                + (f" ({self.size} cells)" if self.size != (r1 - r0 + 1) * (c1 - c0 + 1) else "")
                + f", {rows}, {cols}" + (", large" if self.large else ""))


def objects(board, live, background, large_cells=LARGE_CELLS):
    """Every object on one board, in reading order of each one's first cell."""
    valid = live & (board != background)
    lab = label_components(board, valid).ravel()
    flat = board.ravel()
    idx = np.flatnonzero(lab >= 0)
    if not len(idx):
        return []
    order = np.argsort(lab[idx], kind="stable")
    idx = idx[order]
    _, starts = np.unique(lab[idx], return_index=True)
    return [Obj(flat[cells[0]], cells, large_cells) for cells in np.split(idx, starts[1:])]


def _runs(mask):
    """The horizontal runs of a 64x64 bool mask, in reading order: (row, first col, last col)."""
    p = np.zeros((GRID, GRID + 2), np.int8)
    p[:, 1:-1] = mask
    d = np.diff(p, axis=1)
    rows, c0 = np.nonzero(d == 1)
    _, c1 = np.nonzero(d == -1)
    return rows.tolist(), c0.tolist(), (c1 - 1).tolist()


def components(mask, start=None):
    """The 4-connected parts of `mask`, each as sorted flat indices, in reading
    order of their first cells; with `start` (a flat index), only the part that
    holds it (an empty list if `mask` does not)."""
    rows, c0, c1 = _runs(mask)
    by_row = [[] for _ in range(GRID)]
    for i, r in enumerate(rows):
        by_row[r].append(i)
    seen, out = [False] * len(rows), []
    if start is None:
        seeds = range(len(rows))
    else:
        y, x = divmod(int(start), GRID)
        seeds = [i for i in by_row[y] if c0[i] <= x <= c1[i]]
    for s in seeds:
        if seen[s]:
            continue
        seen[s], todo, part = True, [s], []
        while todo:
            i = todo.pop()
            part.append(i)
            for r in (rows[i] - 1, rows[i] + 1):
                if 0 <= r < GRID:
                    for j in by_row[r]:
                        if not seen[j] and c0[j] <= c1[i] and c1[j] >= c0[i]:
                            seen[j] = True
                            todo.append(j)
        part.sort()
        out.append(np.concatenate([np.arange(rows[i] * GRID + c0[i], rows[i] * GRID + c1[i] + 1) for i in part]))
    return out


def subject(board, action, live, background, mover, large_cells=LARGE_CELLS):
    """The object a move is about, or None. A click (action >= 5): the object
    under the clicked cell (None on the background or a decoration cell). A
    button: the largest object of the `mover` colour; a tie goes to the one
    whose first cell comes first in reading order (None without a mover)."""
    board = np.asarray(board)
    if action >= 5:
        cell = action - 5
        c = int(board.flat[cell])
        if not live.flat[cell] or c == background:
            return None
        part = components(live & (board == c), cell)
        return Obj(c, part[0], large_cells)
    if mover is None or mover == background:
        return None
    best = None
    for cells in components(live & (board == mover)):
        if best is None or len(cells) > len(best):
            best = cells
    return None if best is None else Obj(mover, best, large_cells)


# ---- relations -----------------------------------------------------------------------------------
def sits_on(board, obj, live, background):
    """The colour an object sits on: the commonest colour of the live cells
    around it (4-neighbours outside it), the background on a tie or when there
    are none. (Until 16 Oct a tie went to the lower colour number, so a mover
    between two walls numbered below the floor "sat on" the walls and touched
    nothing; S1 VERSION 3. Still a limit: in a dead end the walls outnumber the
    floor around it, and it sits on them.)"""
    m = np.zeros(GRID * GRID, bool)
    m[obj.cells] = True
    m = m.reshape(GRID, GRID)
    ring = np.zeros_like(m)
    ring[1:] |= m[:-1]
    ring[:-1] |= m[1:]
    ring[:, 1:] |= m[:, :-1]
    ring[:, :-1] |= m[:, 1:]
    ring &= ~m & live
    if not ring.any():
        return background
    counts = np.bincount(board[ring], minlength=16)
    return background if counts[background] == counts.max() else int(counts.argmax())


def relations(board, obj, live, background, ray_max=RAY_MAX):
    """What is next to an object, per direction: {dir: {dist, colour, touch, edge}}.
    `dist`: cells from its edge to the first live cell that is neither part of
    it nor the colour it sits on (or the background); None if there is none
    before the screen edge or `ray_max`. `edge`: cells from its edge to the
    screen edge (0 = at the edge). Also returns the colour it sits on."""
    under = sits_on(board, obj, live, background)
    ys, xs = obj.cells // GRID, obj.cells % GRID
    own = np.zeros((GRID, GRID), bool)
    own[ys, xs] = True
    blocking = live & ~own & (board != under) & (board != background)
    out = {}
    for d, (dy, dx) in DIRS.items():
        best = None
        lines = np.unique(ys) if dy == 0 else np.unique(xs)
        for ln in lines:                                   # each row (or column) the object faces along
            if dy == 0:
                cols = xs[ys == ln]
                x0 = cols.max() if dx > 0 else cols.min()
                ray = blocking[ln, x0 + 1:] if dx > 0 else blocking[ln, :x0][::-1]
                vals = board[ln, x0 + 1:] if dx > 0 else board[ln, :x0][::-1]
            else:
                rows = ys[xs == ln]
                y0 = rows.max() if dy > 0 else rows.min()
                ray = blocking[y0 + 1:, ln] if dy > 0 else blocking[:y0, ln][::-1]
                vals = board[y0 + 1:, ln] if dy > 0 else board[:y0, ln][::-1]
            ray = ray[:ray_max]
            if ray.any():
                i = int(np.argmax(ray))
                if best is None or i + 1 < best[0]:
                    best = (i + 1, int(vals[i]))
        edge = {"up": int(ys.min()), "down": GRID - 1 - int(ys.max()),
                "left": int(xs.min()), "right": GRID - 1 - int(xs.max())}[d]
        out[d] = {"dist": best[0] if best else None, "colour": best[1] if best else None,
                  "touch": bool(best and best[0] == 1), "edge": edge}
    return out, under


def inside(obj, objs):
    """The smallest larger object whose box contains this object's box, or None."""
    r0, r1, c0, c1 = obj.box
    boxes = [o for o in objs if o is not obj and o.size > obj.size and o.box[0] <= r0 and o.box[1] >= r1
             and o.box[2] <= c0 and o.box[3] >= c1]
    return min(boxes, key=lambda o: o.size) if boxes else None


# ---- feature bits --------------------------------------------------------------------------------
def feature_names(colours):
    """The fixed list of yes/no facts for a level whose screens use `colours`."""
    names = [f"action{a}" for a in range(1, 8)]
    names += ["subject", "subject_large", "subject_inside", "subject_on_background", "clicked_background"]
    names += [f"subject_size<={k}" for k in SIZE_BITS] + [f"subject_colour={c}" for c in colours]
    for d in DIRS:
        names += [f"{d}_touch", f"{d}_edge", f"{d}_clear"] + [f"{d}_ray<={k}" for k in RAY_BITS]
        names += [f"{d}_colour={c}" for c in colours]
    return names


def feature_set(action, subject, rels, inside_obj, under, background, clicked_bg):
    """The names of one case's facts that are true."""
    on = {f"action{action + 1 if action < 5 else 6}"}
    if clicked_bg:
        on.add("clicked_background")
    if subject is not None:
        on.add("subject")
        on.update(f"subject_size<={k}" for k in SIZE_BITS if subject.size <= k)
        on.add(f"subject_colour={subject.colour}")
        if subject.large:
            on.add("subject_large")
        if inside_obj is not None:
            on.add("subject_inside")
        if under == background:
            on.add("subject_on_background")
        for d, r in rels.items():
            if r["touch"]:
                on.add(f"{d}_touch")
            if r["edge"] == 0:
                on.add(f"{d}_edge")
            if r["dist"] is None:
                on.add(f"{d}_clear")
            else:
                on.update(f"{d}_ray<={k}" for k in RAY_BITS if r["dist"] <= k)
                on.add(f"{d}_colour={r['colour']}")
    return on


def feature_row(names, action, subject, rels, inside_obj, under, background, clicked_bg):
    """One case's yes/no facts, in the order of `names`."""
    on = feature_set(action, subject, rels, inside_obj, under, background, clicked_bg)
    return np.array([n in on for n in names], bool)


# ---- in the rule sandbox -------------------------------------------------------------------------
def features(board, act, api, mover=None, with_inside=False):
    """The facts S1 records for a move that are true (feature_set), worked out
    from what a rule is given: the board, the action, the api, and the mover
    colour (a PARAMS slot; None for a level without one). `subject_inside` needs
    every object on the board, so it is only worked out `with_inside`.

        f = features(board, act, api, p["mover"])
        if "left_touch" in f: ...             # the mover is touching something on its left
    """
    board = np.asarray(board)
    live = ~np.asarray(api.ticker, bool)
    bg = api.background
    a = act.action - 1 if act.click is None else 5 + act.click[0] * GRID + act.click[1]
    s = subject(board, a, live, bg, mover)
    rels, under, ins = {}, bg, None
    if s is not None:
        rels, under = relations(board, s, live, bg)
        if with_inside:
            ins = inside(s, objects(board, live, bg))
    return feature_set(a, s, rels, ins, under, bg, a >= 5 and s is None)


# ---- H1's primitives: what rules may call ---------------------------------------------------------
# Every primitive takes the board and `api` first, never changes its inputs, never raises on a
# missing object (it returns the board unchanged with nothing claimed), and returns either an
# answer or (next board, claimed cells), so a rule can return a primitive's result directly.
def _live(api):
    return ~np.asarray(api.ticker, bool)


def _nothing(board):
    return np.array(board), np.zeros((GRID, GRID), bool)


def _cells(x):
    """A 64x64 bool mask from an object or a mask."""
    return x.mask if isinstance(x, Obj) else np.asarray(x, bool)


def rule_objects(board, api, colour=None):
    """The objects on the board (4-connected one-colour areas, not the background
    or ticker cells), largest first; only those of `colour` if given.
    Each has .colour, .size, .box (top, bottom, left, right) and .mask.
        objs = objects(board, api, p["wall"])"""
    board = np.asarray(board)
    found = objects(board, _live(api), api.background)
    if colour is not None:
        found = [o for o in found if o.colour == colour]
    return sorted(found, key=lambda o: -o.size)


def find_object(board, api, colour):
    """The largest object of `colour`, or None.
        me = find_object(board, api, p["agent"])"""
    board = np.asarray(board)
    if colour is None or colour == api.background:
        return None
    return subject(board, 0, _live(api), api.background, colour)


def object_at(board, api, rc):
    """The object under cell rc = (row, col), e.g. the clicked one, or None.
        hit = object_at(board, api, act.click)"""
    if rc is None:
        return None
    y, x = int(rc[0]), int(rc[1])
    if not (0 <= y < GRID and 0 <= x < GRID):
        return None
    return subject(np.asarray(board), 5 + y * GRID + x, _live(api), api.background, None)


def ray(board, api, rc, dy, dx):
    """From cell rc, step by (dy, dx) to the first live cell that is not the
    background: (steps taken, its colour), or (None, None) at the screen edge.
        dist, colour = ray(board, api, (5, 7), 0, 1)"""
    board, live = np.asarray(board), _live(api)
    y, x = int(rc[0]) + dy, int(rc[1]) + dx
    n = 1
    while (dy or dx) and 0 <= y < GRID and 0 <= x < GRID:
        if live[y, x] and board[y, x] != api.background:
            return n, int(board[y, x])
        y, x, n = y + dy, x + dx, n + 1
    return None, None


def _shifted(mask, dy, dx):
    """The mask moved by (dy, dx), or None if any cell would leave the screen."""
    ys, xs = np.nonzero(mask)
    if not len(ys) or ys.min() + dy < 0 or ys.max() + dy >= GRID or xs.min() + dx < 0 or xs.max() + dx >= GRID:
        return None
    out = np.zeros_like(mask)
    out[ys + dy, xs + dx] = True
    return out


def is_blocked(board, api, obj, dy, dx, blockers=None):
    """Would moving obj by (dy, dx) put a cell off the screen, or onto a cell
    (not its own) that is not the background, or, with `blockers` (a list of
    colours), onto one of those colours? True if obj is None.
        if is_blocked(board, api, me, 0, -1): ..."""
    if obj is None:
        return True
    board, m = np.asarray(board), _cells(obj)
    to = _shifted(m, dy, dx)
    if to is None:
        return True
    land = board[to & ~m]
    return bool(np.isin(land, blockers).any() if blockers is not None else (land != api.background).any())


def translate(board, api, obj, dy, dx, behind=None):
    """obj moved by (dy, dx): its old cells become `behind` (default: the
    background), its new cells its colour. Claims the old and new cells.
    Off the screen: the board unchanged, nothing claimed.
        return translate(board, api, me, -1, 0)"""
    if obj is None:
        return _nothing(board)
    board, m = np.asarray(board), _cells(obj)
    to = _shifted(m, dy, dx)
    if to is None:
        return _nothing(board)
    out = np.array(board)
    colours = board[m]
    out[m] = api.background if behind is None else behind
    ys, xs = np.nonzero(m)
    out[ys + dy, xs + dx] = colours                      # keeps a many-colour object's pattern
    return out, m | to


def slide(board, api, obj, dy, dx, blockers=None, behind=None):
    """obj moved step by step by (dy, dx) until the next step is_blocked. Claims
    the old and new cells (only the old ones if it cannot move at all).
        return slide(board, api, me, 0, 1)"""
    if obj is None:
        return _nothing(board)
    board, m = np.asarray(board), _cells(obj)
    if not (dy or dx):
        return np.array(board), np.array(m)
    sy = sx = 0
    cur, clear = m, _without(board, m, api, behind)      # it never blocks itself
    while not is_blocked(clear, api, cur, dy, dx, blockers):
        cur = _shifted(cur, dy, dx)
        sy, sx = sy + dy, sx + dx
    if not (sy or sx):
        return np.array(board), np.array(m)
    return translate(board, api, m, sy, sx, behind)


def _without(board, m, api, behind):
    """The board with an object's own cells cleared."""
    out = np.array(board)
    out[m] = api.background if behind is None else behind
    return out


def recolour(board, api, cells, colour):
    """`cells` (an object or a 64x64 bool mask) painted `colour`; claims them.
        return recolour(board, api, object_at(board, api, act.click), p["paint"])"""
    if cells is None:
        return _nothing(board)
    m = _cells(cells)
    out = np.array(board)
    out[m] = colour
    return out, np.array(m)


def swap(board, api, a, b):
    """The contents of two same-sized cell sets (objects or masks) exchanged, cell
    by cell in reading order; claims both. Different sizes: the board unchanged,
    nothing claimed.
        return swap(board, api, me, neighbour)"""
    if a is None or b is None:
        return _nothing(board)
    ma, mb = _cells(a), _cells(b)
    if ma.sum() != mb.sum() or (ma & mb).any():
        return _nothing(board)
    board = np.asarray(board)
    out = np.array(board)
    out[ma], out[mb] = board[mb], board[ma]
    return out, ma | mb


def unchanged(board, cells):
    """The board as it is, claiming `cells` (an object or a mask): "these cells
    stay as they are", for no-op and exception rules.
        return unchanged(board, me)"""
    if cells is None:
        return _nothing(board)
    return np.array(board), np.array(_cells(cells))


PRIMITIVES = {"objects": rule_objects, "find_object": find_object, "object_at": object_at, "ray": ray,
              "is_blocked": is_blocked, "translate": translate, "slide": slide, "recolour": recolour,
              "swap": swap, "unchanged": unchanged}

CARD = """Helpers you may call (already defined; tested). They never change their inputs, and
return None or an unchanged board with nothing claimed when an object is missing, so
check for None where it matters. An object has .colour, .size, .box = (top, bottom,
left, right) inclusive, and .mask (64x64 bool, True on its cells).

  objects(board, api, colour=None) -> objects (4-connected one-colour areas, not the
      background or ticker cells), largest first; only `colour` if given
  find_object(board, api, colour) -> the largest object of that colour, or None
  object_at(board, api, (row, col)) -> the object under that cell, or None
  ray(board, api, (row, col), dy, dx) -> (steps, colour) of the first non-background
      cell in that direction, or (None, None) at the screen edge
  is_blocked(board, api, obj, dy, dx, blockers=None) -> True if moving obj by (dy, dx)
      would leave the screen or land on a non-background cell (or on a colour in the
      list `blockers`)
  translate(board, api, obj, dy, dx, behind=None) -> (next_board, claimed): obj moved
      by (dy, dx), its old cells set to `behind` (default api.background); claims old
      and new cells
  slide(board, api, obj, dy, dx, blockers=None, behind=None) -> (next_board, claimed):
      obj moved by (dy, dx) again and again until blocked
  recolour(board, api, cells, colour) -> (next_board, claimed): an object or mask painted
  swap(board, api, a, b) -> (next_board, claimed): two same-sized objects or masks
      exchange contents
  unchanged(board, cells) -> (board, claimed): "these cells stay as they are"

Each (next_board, claimed) result can be returned from predict() as it is.

    def predict(board, act, api, p):
        me = find_object(board, api, p["agent"])
        if is_blocked(board, api, me, -p["step"], 0):
            return unchanged(board, me)
        return translate(board, api, me, -p["step"], 0)"""


SANDBOX = {"features": features, **PRIMITIVES}
