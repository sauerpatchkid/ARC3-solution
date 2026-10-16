"""s1_see.py - S1 SEE: every move as objects, what happened to each, and whether that explains the move.

The problem it fixes: the LLM has to work out from pictures what moved and what
blocked it, and when a rule fails we can't tell whether the model couldn't SEE
the mechanism or couldn't WRITE it. S1 describes each change as objects, with no
LLM, and measures how much of the change that description explains: where it
is low, perception is the bottleneck, not the LLM. (Plan section 5.1.)

All eight steps of the plan, built 16 Oct (1-3 first, then 4-8):

  1. OBJECTS. 4-connected single-colour regions of live cells that are not the
     background colour (gridtools.label_components). Regions over `large_cells`
     are kept and flagged `large` (walls and floors).
  2. CORRESPONDENCE. The objects whose cells changed, before and after, are
     paired in this order: same cells, new colour -> RECOLOURED; same colour
     and shape somewhere else -> MOVED (smallest displacement first); same
     colour, overlapping cells (most overlap first) -> GREW if the old cells are
     all still there, SHRANK if every new cell was already there, else
     RESHAPED. What is left is VANISHED (before) or APPEARED (after). An object
     whose counterpart has no changed cell (a bar that got shorter) is paired too.
  3. CLOSURE. Erase every before-object that went (moved, grew, shrank,
     reshaped, vanished) to the colour it LEFT BEHIND (the commonest colour the
     after-screen shows in the cells it vacated, other than cells now taken by
     an object on top of it: one that moved, recoloured or appeared, or any
     smaller object, since small objects sit on large ones; the text says
     "leaving <colour>"), paint every after-object (moved,
     recoloured, grew, shrank, reshaped, appeared), and compare with the real
     after-screen on live cells.
       closure    the whole description rebuilds the screen. It fails only
                  where an object leaves behind more than one colour (it covered
                  a pattern): one layer per object cannot say what was under it.
                  Otherwise RESHAPED and APPEARED make it true by construction,
                  so it is mostly a check on S1 itself (the plan's step 3).
       unit       the share of changed cells rebuilt without the free-form
                  RESHAPED items. The informative number: an object moving over a
                  floor that is not the background, a merge or a morph need more
                  than whole objects that move, recolour, grow, shrink, vanish or
                  appear.
       A floor that only changed because something moved across it adds
       nothing: a grew/shrank/reshaped item is dropped (counted as `absorbed`)
       if the screen still rebuilds without it and it is large or its colour is
       what a moved or vanished object left behind.
       explained  closure, unit == 1 and at most `compact_items` items: the
                  change is a short list of whole-object operations. The
                  funnel's "explained by S1". (A Glassbox refinement of the plan's
                  "closure", measured 16 Oct; G1's use of it is settled in its
                  pre-registration.)

  4. RELATIONS, for the SUBJECT of each move: the mover for a button (step 6),
     the clicked object for a click. In each direction: the distance from its
     edge to the first cell that is neither it nor what it sits on, that cell's
     colour (touching = 1 cell away), and the distance to the screen edge. Also
     whether it is inside a larger object. A moved object also gets the same
     facts AHEAD of it, in the direction it moved: these explain blocked moves.
  5. FEATURE BITS: those relations as a fixed list of yes/no facts per case
     (`<dir>_touch`, `<dir>_ray<=4`, `<dir>_colour=9`, `subject_size<=4`, ...).
     S3's guard fitter learns when a rule fails from these, aligning a
     direction to the rule's own move.
  6. ROLES, per level, from all its cases: the MOVER (a colour whose objects
     move under at least `mover_min_buttons` buttons, `mover_min_moves` moves;
     most moves first, a tie to the colour with the fewest objects on the
     level's first screen),
     STATIC colours (never change), CLICK-CHANGERS and CLICK-NOOPS (colours
     whose clicks change something / never do), COUNTERS (objects at the
     screen's edge that only grow or shrink: bars). Hints for bind() (S4).
  7. EXPLORATION MATRIX, per level: for each (subject type = colour, size
     bucket, shape; action family), how often each outcome happened (nothing,
     moved, recoloured, vanished, appeared, other). Mixed outcomes (the row's
     normalised entropy) are OPINE-World's "ontology error": S5 probes there.
  8. TEXT: what the LLM will read beside each move's pictures, at most
     `text_lines` lines. A FIRST DRAFT, pinned by a golden test once S2 uses it.

Reads  <run>/evidence/<game>_L<k>_<split>.npz
Writes <run>/s1_see/<game>_L<k>_<split>.facts.npz    per case: counts by kind, closure, unit, explained,
                                                      subject, distances, feature bits (+ their names)
       <run>/s1_see/<game>_L<k>_<split>.facts.jsonl  per case: the object diff item by item, and the text
       <run>/s1_see/<game>_L<k>_<split>.roles.json   step 6
       <run>/s1_see/<game>_L<k>_<split>.matrix.json  step 7
       <run>/s1_see/<game>.summary.json              per level and split: the closure numbers
"""
import collections
import hashlib
import json
import math
import os

import numpy as np

from gb import artifacts as A
from gb.core.evidence import PALETTE_NAMES, LevelEvidence
from gridtools import GRID, border_band, label_components

VERSION = 2                     # bump when what the output means changes (artifacts.py); 2 adds steps 4-8
KINDS = ("moved", "recoloured", "grew", "shrank", "vanished", "appeared", "reshaped")
UNIT_KINDS = ("moved", "recoloured", "grew", "shrank", "vanished", "appeared")     # all but free-form reshapes
ERASE = ("moved", "grew", "shrank", "reshaped", "vanished")                       # the before-object goes
PAINT = ("moved", "recoloured", "grew", "shrank", "reshaped", "appeared")         # the after-object is drawn
DEFAULTS = {"connectivity": 4, "large_cells": 1024, "compact_items": 8, "ray_max": 64, "text_lines": 12,
            "mover_min_buttons": 2, "mover_min_moves": 10, "matrix_size_buckets": [1, 4, 16, 64]}
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OUTCOMES = ("nothing", "moved", "recoloured", "vanished", "appeared", "other")


# ---- 1. objects -------------------------------------------------------------------------------
class Obj:
    """One single-colour region: its colour, its cells (flat indices), box and shape."""
    __slots__ = ("colour", "cells", "size", "box", "shape", "large")

    def __init__(self, colour, cells, large_cells):
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


def objects(board, live, background, large_cells=DEFAULTS["large_cells"]):
    """The objects on one board."""
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


# ---- 2. correspondence -------------------------------------------------------------------------
def correspond(before_objs, after_objs, changed):
    """Pair the objects a change touched. Returns items: dicts with `kind`,
    `before` and/or `after` (Obj) and, for a move, (dy, dx)."""
    bs = [o for o in before_objs if changed[o.cells].any()]
    as_ = [o for o in after_objs if changed[o.cells].any()]
    # An object can change while its counterpart on the other board has no changed
    # cell (a bar that shrinks: the shorter bar's cells are all unchanged). So bring
    # in every same-colour object on the other board that overlaps a touched one.
    def under(pool, touched, have):
        return [o for o in pool if o not in have and any(
            t.colour == o.colour and len(np.intersect1d(t.cells, o.cells, assume_unique=True)) for t in touched)]
    extra_b, extra_a = under(before_objs, as_, bs), under(after_objs, bs, as_)
    bs, as_ = bs + extra_b, as_ + extra_a
    used_b, used_a, items = set(), set(), []

    def take(i, j, **item):
        used_b.add(i)
        used_a.add(j)
        items.append(dict(item, before=bs[i], after=as_[j]))

    same_cells = {o.cells.tobytes(): j for j, o in enumerate(as_)}
    for i, x in enumerate(bs):                                   # recoloured: the same cells, a new colour
        j = same_cells.get(x.cells.tobytes())
        if j is not None and as_[j].colour != x.colour and j not in used_a:
            take(i, j, kind="recoloured")
    moves = []
    for i, x in enumerate(bs):                                   # moved: same colour and shape elsewhere
        if i in used_b:
            continue
        for j, y in enumerate(as_):
            if j not in used_a and y.colour == x.colour and y.shape == x.shape:
                dy, dx = y.box[0] - x.box[0], y.box[2] - x.box[2]
                if (dy, dx) != (0, 0):
                    moves.append((abs(dy) + abs(dx), x.box, i, j, dy, dx))
    for _, _, i, j, dy, dx in sorted(moves):                     # smallest displacement first
        if i not in used_b and j not in used_a:
            take(i, j, kind="moved", dy=dy, dx=dx)
    overlaps = []
    for i, x in enumerate(bs):                                   # reshaped: same colour, overlapping
        if i in used_b:
            continue
        for j, y in enumerate(as_):
            if j not in used_a and y.colour == x.colour:
                n = len(np.intersect1d(x.cells, y.cells, assume_unique=True))
                if n:
                    overlaps.append((-n, x.box, i, j))
    for n, _, i, j in sorted(overlaps):                          # most overlap first
        if i not in used_b and j not in used_a:
            kind = "grew" if -n == bs[i].size else "shrank" if -n == as_[j].size else "reshaped"
            take(i, j, kind=kind)
    items += [{"kind": "vanished", "before": x} for i, x in enumerate(bs) if i not in used_b]
    items += [{"kind": "appeared", "after": y} for j, y in enumerate(as_) if j not in used_a]
    return items


# ---- 3. closure ----------------------------------------------------------------------------------
def left_behind(after, item, taken, background):
    """The colour an object leaves behind: the commonest colour of the after-
    board in the cells it vacated (its old cells that are not its own new cells),
    other than `taken` cells (now occupied by an object that moved, recoloured or
    appeared); the background if none are left."""
    obj = item["before"]
    gone = ~taken[obj.cells]
    if "after" in item:
        gone &= ~np.isin(obj.cells, item["after"].cells, assume_unique=True)
    vac = obj.cells[gone]
    if not len(vac):
        return background
    return int(np.bincount(after.ravel()[vac], minlength=16).argmax())


def rebuild(before, items, kinds=KINDS):
    """The before-board with the items of `kinds` applied: erase each departing
    object to what it sat on, then paint the after-objects."""
    out = before.ravel().copy()
    for it in items:
        if it["kind"] in kinds and it["kind"] in ERASE:
            out[it["before"].cells] = it["under"]
    for it in items:
        if it["kind"] in kinds and it["kind"] in PAINT:
            out[it["after"].cells] = it["after"].colour
    return out.reshape(before.shape)


def see_case(before, after, live, background, settings=DEFAULTS, objs=None):
    """One case: its object diff and how much of the change it explains.
    `objs(board)`: the objects of a board (see_level passes a cached one)."""
    if objs is None:
        objs = lambda board: objects(board, live, background, settings["large_cells"])
    changed = ((before != after) & live).ravel()
    n_changed = int(changed.sum())
    if not n_changed:
        return {"changed": 0, "items": [], "closure": True, "unit": 1.0, "explained": True, "n_objects": 0,
                "absorbed": 0}
    before_objs = objs(before)
    items = correspond(before_objs, objs(after), changed)
    for it in items:
        if "before" in it:                             # cells now held by something on top of it
            taken = np.zeros(GRID * GRID, bool)
            for other in items:
                if other is not it and "after" in other and (
                        other["kind"] in ("moved", "recoloured", "appeared")
                        or other["after"].size < it["before"].size):
                    taken[other["after"].cells] = True
            it["under"] = left_behind(after, it, taken, background)
    a, lv = after.ravel(), live.ravel()
    exact = lambda its: bool((rebuild(before, its).ravel()[lv] == a[lv]).all())
    absorbed = 0
    left = {it["under"] for it in items if it["kind"] in ("moved", "vanished")}
    for it in [it for it in items if it["kind"] in ("grew", "shrank", "reshaped")
               and (it["before"].large or it["before"].colour in left)]:
        rest = [x for x in items if x is not it]
        if exact(rest):                       # a floor that only changed because something crossed it
            items, absorbed = rest, absorbed + 1
    full = rebuild(before, items).ravel()
    unit = rebuild(before, items, UNIT_KINDS).ravel()
    closure = bool((full[lv] == a[lv]).all())
    unit_share = float(((unit == a) & changed).sum()) / n_changed
    return {"changed": n_changed, "items": items, "closure": closure, "unit": unit_share, "absorbed": absorbed,
            "explained": closure and unit_share == 1.0 and len(items) <= settings["compact_items"],
            "n_objects": len(before_objs)}


def item_text(it):
    """One item in plain words (a first draft of the facts text; not yet frozen)."""
    k = it["kind"]
    leaving = f", leaving colour {it['under']} {PALETTE_NAMES[it['under']]}" if "under" in it else ""
    if k == "moved":
        return f"object ({it['before'].describe()}) moved ({it['dy']:+d}, {it['dx']:+d}){leaving}"
    if k == "vanished":
        return f"object ({it['before'].describe()}) vanished{leaving}"
    if k == "recoloured":
        return f"object ({it['before'].describe()}) recoloured {it['before'].colour} -> {it['after'].colour}"
    if k in ("grew", "shrank", "reshaped"):
        verb = {"grew": "grew", "shrank": "shrank", "reshaped": "changed shape"}[k]
        return (f"object ({it['before'].describe()}) {verb}: {it['before'].size} -> "
                f"{it['after'].size} cells, now ({it['after'].describe()})")
    return f"object ({(it.get('before') or it.get('after')).describe()}) {k}"


def item_record(it):
    o = it.get("before") or it.get("after")
    rec = {"kind": it["kind"], "colour": o.colour, "size": o.size, "box": list(o.box), "large": o.large,
           "text": item_text(it)}
    if it["kind"] == "moved":
        rec.update(dy=it["dy"], dx=it["dx"])
    if "under" in it:
        rec["left"] = it["under"]
    if it["kind"] in ("recoloured", "grew", "shrank", "reshaped"):
        rec.update(colour_to=it["after"].colour, size_to=it["after"].size, box_to=list(it["after"].box))
    return rec


# ---- 4. relations ---------------------------------------------------------------------------------
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


def relations(board, obj, live, background, ray_max=64):
    """What is next to an object, per direction: {dir: {dist, colour, touch, edge}}.
    `dist`: cells from its edge to the first live cell that is neither part of
    it nor the colour it sits on (or the background); None if there is none
    before the screen edge or `ray_max`. `edge`: cells from its edge to the
    screen edge (0 = at the edge)."""
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


def direction_of(dy, dx):
    """The direction a move goes in, by its larger component."""
    return ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")


# ---- 5. feature bits -------------------------------------------------------------------------------
SIZE_BITS = (1, 4, 16, 64)
RAY_BITS = (2, 4, 8)


def feature_names(colours):
    """The fixed list of yes/no facts for a level whose screens use `colours`."""
    names = [f"action{a}" for a in range(1, 8)]
    names += ["subject", "subject_large", "subject_inside", "subject_on_background", "clicked_background"]
    names += [f"subject_size<={k}" for k in SIZE_BITS] + [f"subject_colour={c}" for c in colours]
    for d in DIRS:
        names += [f"{d}_touch", f"{d}_edge", f"{d}_clear"] + [f"{d}_ray<={k}" for k in RAY_BITS]
        names += [f"{d}_colour={c}" for c in colours]
    return names


def feature_row(names, action, subject, rels, inside_obj, under, background, clicked_bg):
    """One case's yes/no facts, in the order of `names`."""
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
    return np.array([n in on for n in names], bool)


# ---- 6. roles -------------------------------------------------------------------------------------
def roles(ev, results, settings=DEFAULTS):
    """Role hints for one level, from all its cases (module doc, step 6)."""
    moves = collections.defaultdict(collections.Counter)      # colour -> {button: recorded moves}
    involved, counters = set(), collections.Counter()
    band = border_band()
    for k, r in results.items():
        a = int(ev.actions[k])
        for it in r["items"]:
            for side in ("before", "after"):
                if side in it:
                    involved.add(it[side].colour)
            o = it.get("before") or it.get("after")
            if it["kind"] == "moved" and a < 5:
                moves[o.colour][a] += int(ev.count[k])
            if it["kind"] in ("grew", "shrank") and band.ravel()[o.cells].all():
                counters[o.colour] += 1
    clicks, click_changed = collections.Counter(), collections.Counter()
    for k in np.flatnonzero(ev.actions >= 5):
        y, x = divmod(int(ev.actions[k]) - 5, GRID)
        c = int(ev.before[k][y, x])
        clicks[c] += int(ev.count[k])
        if ev.changed[k]:
            click_changed[c] += int(ev.count[k])
    first = objects(ev.first, ev.live, ev.api.background, settings["large_cells"])
    count_on_first = collections.Counter(o.colour for o in first)
    movers = sorted((c for c, b in moves.items()                   # most moves; a tie goes to the colour with
                     if len(b) >= settings["mover_min_buttons"]       # the fewest objects on the first screen
                     and sum(b.values()) >= settings["mover_min_moves"]),   # (a sprite you steer is unique)
                    key=lambda c: (-sum(moves[c].values()), count_on_first[c], c))
    present = sorted(int(c) for c in np.unique(ev.before[:, ev.live]))
    return {"mover": movers[0] if movers else None, "movers": movers,
            "static": [c for c in present if c not in involved],
            "click_changers": sorted(c for c in clicks if click_changed[c]),
            "click_noops": sorted(c for c in clicks if not click_changed[c]),
            "counters": sorted(c for c, n in counters.items() if n >= 3),
            "colours": present,
            "evidence": {"button_moves": {str(c): {f"ACTION{a + 1}": n for a, n in sorted(b.items())}
                                          for c, b in moves.items()},
                         "clicks": {str(c): [click_changed[c], n] for c, n in sorted(clicks.items())}}}


# ---- 7. exploration matrix ------------------------------------------------------------------------
def object_type(o, buckets):
    """An object's type: colour, size bucket and shape."""
    lo = 1
    for b in buckets:
        if o.size <= b:
            size = f"{lo}" if lo == b else f"{lo}-{b}"
            break
        lo = b + 1
    else:
        size = f"{lo}+"
    shape = hashlib.sha1(repr(o.shape).encode()).hexdigest()[:6]
    return f"colour {o.colour} {PALETTE_NAMES[o.colour]}, {size} cells, shape {shape}"


def outcome_of(subject, items):
    """What happened to the subject in a case: one of OUTCOMES."""
    for it in items:
        if it.get("before") is subject:
            return it["kind"] if it["kind"] in OUTCOMES else "other"
    return "nothing"


def matrix(rows):
    """{(type, family): Counter of outcomes} -> sorted rows with their entropy."""
    out = []
    for (typ, fam), c in rows.items():
        n = sum(c.values())
        h = max(0.0, -sum(v / n * math.log(v / n) for v in c.values() if v) / math.log(len(OUTCOMES)))
        out.append({"type": typ, "family": fam, "cases": n, "counts": {o: c[o] for o in OUTCOMES if c[o]},
                    "entropy": round(h, 3)})
    return sorted(out, key=lambda r: (-r["entropy"], -r["cases"], r["type"], r["family"]))


# ---- 8. text --------------------------------------------------------------------------------------
def colour_name(c):
    return f"colour {c} {PALETTE_NAMES[c]}"


def relation_text(d, r):
    if r["touch"]:
        return f"{d}: touching {colour_name(r['colour'])}"
    if r["dist"] is not None:
        return f"{d}: {colour_name(r['colour'])} {r['dist']} cells away"
    return f"{d}: at the screen edge" if r["edge"] == 0 else f"{d}: clear to the screen edge ({r['edge']} cells)"


def facts_text(action, subject, rels, inside_obj, clicked_colour, r, aheads, others, max_lines=12):
    """The facts the LLM will read beside one move's pictures (a first draft)."""
    if action < 5:
        head = [f"ACTION{action + 1}."]
    else:
        y, x = divmod(action - 5, GRID)
        on = f"an object ({subject.describe()})" if subject else f"the background ({colour_name(clicked_colour)})"
        head = [f"Click at row {y}, col {x} on {on}."]
        if subject is not None and inside_obj is not None:
            head.append(f"  It is inside a larger object ({inside_obj.describe()}).")
    if subject is not None and action < 5:
        head.append(f"  The mover ({subject.describe()}): " + "; ".join(relation_text(d, rels[d]) for d in DIRS) + ".")
    items = (r or {}).get("items") or []
    if not items:
        return head + ["  Nothing changed."]
    tail = []
    for it, d, rel in aheads[:2]:
        tail.append(f"  Ahead of the {colour_name(it['before'].colour)} object that moved {d}: "
                    + relation_text(d, rel).split(": ", 1)[1] + ".")
    explained = ("The description explains the whole change." if r["explained"] else
                 f"Whole objects explain {r['unit']:.0%} of the changed cells.")
    tail.append(f"  {others} other object{'' if others == 1 else 's'} did not change. {explained}")
    room = max(1, max_lines - len(head) - len(tail))
    body = ["  " + item_text(it)[0].upper() + item_text(it)[1:] + "." for it in items]
    if len(body) > room:
        body = body[:room - 1] + [f"  ... and {len(body) - room + 1} more changes."]
    return head + body + tail


# ---- one level, one run -------------------------------------------------------------------------------
def see_level(ev, settings=DEFAULTS):
    """Every case of one level's evidence. Pass 1 describes each change (steps
    1-3); the level's roles follow from all of them (6); pass 2 gives each case
    its subject, relations, feature bits, matrix row and text (4, 5, 7, 8).
    Returns (arrays per case, records per case, roles, matrix)."""
    if settings.get("connectivity", 4) != 4:
        raise NotImplementedError("S1 groups cells 4-connected; 8-connected objects are not built")
    n = len(ev.actions)
    arr = {"changed_cells": np.zeros(n, np.int32), "items": np.zeros(n, np.int32), "closure": np.ones(n, bool),
           "unit": np.ones(n, np.float32), "explained": np.ones(n, bool), "objects": np.zeros(n, np.int32),
           "absorbed": np.zeros(n, np.int16),
           **{f"n_{k}": np.zeros(n, np.int16) for k in KINDS}}
    background = ev.api.background
    cache = {}                     # many cases share a screen: objects once per distinct board

    def objs(board):
        key = board.tobytes()
        if key not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[key] = objects(board, ev.live, background, settings["large_cells"])
        return cache[key]
    # -- pass 1: what changed (steps 1-3) ---------------------------------------------------------------
    results = {}
    for k in np.flatnonzero(ev.changed):
        k = int(k)
        r = see_case(ev.before[k], ev.after[k], ev.live, background, settings, objs)
        results[k] = r
        arr["changed_cells"][k], arr["items"][k], arr["objects"][k] = r["changed"], len(r["items"]), r["n_objects"]
        arr["closure"][k], arr["unit"][k], arr["explained"][k] = r["closure"], r["unit"], r["explained"]
        arr["absorbed"][k] = r["absorbed"]
        for kind in KINDS:
            arr[f"n_{kind}"][k] = sum(it["kind"] == kind for it in r["items"])
    # -- the level's roles (step 6) ---------------------------------------------------------------------
    role = roles(ev, results, settings)
    names = feature_names(role["colours"])
    feats = np.zeros((n, len(names)), bool)
    arr["subject_colour"] = np.full(n, -1, np.int8)
    arr["subject_size"] = np.zeros(n, np.int32)
    for d in DIRS:
        arr[f"dist_{d}"] = np.full(n, -1, np.int16)
        arr[f"edge_{d}"] = np.full(n, -1, np.int16)
    rel_cache, rows, records = {}, collections.defaultdict(collections.Counter), []
    # -- pass 2: every case's subject, relations, features, matrix row and text (steps 4, 5, 7, 8) ------------
    for k in range(n):
        board, a = ev.before[k], int(ev.actions[k])
        bo = objs(board)
        subject, clicked_colour, clicked_bg = None, None, False
        if a >= 5:
            y, x = divmod(a - 5, GRID)
            clicked_colour = int(board[y, x])
            cell = y * GRID + x
            subject = next((o for o in bo if o.cells[np.searchsorted(o.cells, cell) % o.size] == cell), None)
            clicked_bg = subject is None
        elif role["mover"] is not None:
            movers = [o for o in bo if o.colour == role["mover"]]
            subject = max(movers, key=lambda o: o.size) if movers else None
        rels, under, inside_obj = {}, background, None
        if subject is not None:
            key = (board.tobytes(), subject.cells.tobytes())
            if key not in rel_cache:
                if len(rel_cache) > 4096:
                    rel_cache.clear()
                rl, un = relations(board, subject, ev.live, background, settings["ray_max"])
                rel_cache[key] = (rl, un, inside(subject, bo))
            rels, under, inside_obj = rel_cache[key]
            arr["subject_colour"][k], arr["subject_size"][k] = subject.colour, subject.size
            for d in DIRS:
                arr[f"dist_{d}"][k] = -1 if rels[d]["dist"] is None else rels[d]["dist"]
                arr[f"edge_{d}"][k] = rels[d]["edge"]
            fam = "click" if a >= 5 else f"ACTION{a + 1}"
            rows[(object_type(subject, settings["matrix_size_buckets"]), fam)][
                outcome_of(subject, results[k]["items"] if k in results else [])] += 1
        feats[k] = feature_row(names, a, subject, rels, inside_obj, under, background, clicked_bg)
        r = results.get(k)
        aheads = []
        if r:
            for it in r["items"]:
                if it["kind"] == "moved" and it["before"] is not subject:
                    d = direction_of(it["dy"], it["dx"])
                    aheads.append((it, d, relations(board, it["before"], ev.live, background,
                                                    settings["ray_max"])[0][d]))
        touched = {id(it["before"]) for it in (r or {}).get("items", []) if "before" in it}
        text = facts_text(a, subject, rels, inside_obj, clicked_colour, r, aheads,
                          sum(id(o) not in touched for o in bo), settings["text_lines"])
        rec = {"k": k, "group": ev.group[k], "changed": int(r["changed"]) if r else 0, "text": text}
        if r:
            rec.update(closure=r["closure"], absorbed=r["absorbed"], unit=round(r["unit"], 4),
                       explained=r["explained"], items=[item_record(it) for it in r["items"]])
        records.append(rec)
    arr["features"] = np.packbits(feats, axis=1)
    arr["feature_names"] = np.array(names)
    return arr, records, role, matrix(rows)


def summary(ev, arr, settings=DEFAULTS):
    """The closure numbers for one level: shares of its distinct changing cases,
    and of its changed cells (the plan's "perception coverage")."""
    ch = ev.changed
    n, cells = int(ch.sum()), int(arr["changed_cells"].sum())
    share = lambda m: round(int((m & ch).sum()) / n, 4) if n else 0.0
    unit_cells = float((arr["unit"] * arr["changed_cells"]).sum())
    items = arr["items"][ch]
    return {"changing": n, "changed_cells": cells,
            "closure": share(arr["closure"]), "explained": share(arr["explained"]),
            "unit_cells": round(unit_cells / cells, 4) if cells else 0.0,
            "with_reshaped": share(arr["n_reshaped"] > 0),
            "too_many_items": share(arr["items"] > settings["compact_items"]),
            "items_median": float(np.median(items)) if n else 0.0,
            "items_p90": float(np.percentile(items, 90)) if n else 0.0,
            "kinds": {k: int(arr[f"n_{k}"][ch].sum()) for k in KINDS}}


def run_game(run, game, log=print, force=False):
    """S1 on every evidence split of one game in the run. Skips splits that are
    current (artifacts.py), unless `force`."""
    settings = dict(DEFAULTS, **{k: v for k, v in run.config.get("see", {}).items() if k in DEFAULTS})
    ev_dir = os.path.join(run.dir, "evidence")
    splits = sorted(f[:-4] for f in os.listdir(ev_dir) if f.startswith(f"{game}_L") and f.endswith(".npz"))
    out = run.stage_dir("s1_see")
    p_sum = os.path.join(out, f"{game}.summary.json")
    summ = json.load(open(p_sum)) if os.path.exists(p_sum) else {}
    for name in splits:
        p_ev = os.path.join(ev_dir, name + ".npz")
        if not force and run.is_current("s1_see", name, VERSION, settings, {"evidence": p_ev}) and name in summ:
            log(f"[{name}] S1 is current; skipped" + (" (BUT s1_see.py has changed since: re-run with --force "
                "if the change alters the output, and bump VERSION if it alters what the output means)"
                if run.code_changed("s1_see", name, __file__) else ""))
            continue
        ev = LevelEvidence.load(p_ev)
        arr, records, role, mat = see_level(ev, settings)
        p_npz, p_jsonl = os.path.join(out, name + ".facts.npz"), os.path.join(out, name + ".facts.jsonl")
        p_roles, p_matrix = os.path.join(out, name + ".roles.json"), os.path.join(out, name + ".matrix.json")
        np.savez_compressed(p_npz, **arr)
        with open(p_jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        with open(p_roles, "w") as f:
            json.dump(role, f, indent=1)
        with open(p_matrix, "w") as f:
            json.dump(mat, f, indent=1)
        summ[name] = dict(summary(ev, arr, settings), mover=role["mover"],
                          mixed_rows=sum(r["entropy"] > 0 for r in mat), rows=len(mat))
        run.write_header("s1_see", name, VERSION, settings, {"evidence": p_ev},
                         {"facts": p_npz, "records": p_jsonl, "roles": p_roles, "matrix": p_matrix}, code=__file__)
        s = summ[name]
        log(f"[{name}] {s['changing']} changing cases: closure {s['closure']:.1%}, explained {s['explained']:.1%}, "
            f"whole-object cells {s['unit_cells']:.1%}, with a reshaped object {s['with_reshaped']:.1%}, "
            f"items median {s['items_median']:g} (p90 {s['items_p90']:g}); mover {s['mover']}, "
            f"{s['mixed_rows']} of {s['rows']} matrix rows mixed")
    with open(p_sum, "w") as f:
        json.dump(summ, f, indent=1)
    return summ
