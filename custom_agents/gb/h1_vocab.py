"""h1_vocab.py - H1 VOCAB: the object and relation code that S1, S3's guards and the rules share.

The problem it fixes: S1 tells the LLM what is next to the object a move is
about (touching, a ray, the screen edge), S3's guard fitter learns from those
same facts when a rule fails, and the guard it writes must work them out again
inside the rule sandbox, from the board alone. Three copies of that code would
drift apart, and a guard fitted on one copy would be judged by another. This
file is the one copy. (Plan section 5.7. The rest of H1's primitives, such as
`translate` and `slide`, are added here when H1 is built.)

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
    """One single-colour region: its colour, its cells (flat indices, sorted), box and shape."""
    __slots__ = ("colour", "cells", "size", "box", "shape", "large")

    def __init__(self, colour, cells, large_cells=LARGE_CELLS):
        self.colour, self.cells = int(colour), cells
        self.size = len(cells)
        ys, xs = cells // GRID, cells % GRID
        self.box = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        self.shape = (self.box[1] - self.box[0], self.box[3] - self.box[2],
                      ((ys - self.box[0]) * GRID + (xs - self.box[2])).tobytes())    # cells relative to the box
        self.large = self.size > large_cells

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
    around it (4-neighbours outside it), else the background."""
    m = np.zeros(GRID * GRID, bool)
    m[obj.cells] = True
    m = m.reshape(GRID, GRID)
    ring = np.zeros_like(m)
    ring[1:] |= m[:-1]
    ring[:-1] |= m[1:]
    ring[:, 1:] |= m[:, :-1]
    ring[:, :-1] |= m[:, 1:]
    ring &= ~m & live
    return int(np.bincount(board[ring], minlength=16).argmax()) if ring.any() else background


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


SANDBOX = {"features": features}
