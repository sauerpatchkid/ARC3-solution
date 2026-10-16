"""s1_see.py - S1 SEE: every move as objects, what happened to each, and whether that explains the move.

The problem it fixes: the LLM has to work out from pictures what moved and what
blocked it, and when a rule fails we can't tell whether the model couldn't SEE
the mechanism or couldn't WRITE it. S1 describes each change as objects, with no
LLM, and measures how much of the change that description explains: where it
is low, perception is the bottleneck, not the LLM. (Plan section 5.1.)

Skeleton, 16 Oct: steps 1-3 of the plan (objects, correspondence, closure).
Next: relations, feature bits, role hints, the exploration matrix and the facts
text the LLM will read (steps 4-8).

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

Reads  <run>/evidence/<game>_L<k>_<split>.npz
Writes <run>/s1_see/<game>_L<k>_<split>.facts.npz    per case: counts by kind, closure, unit, explained
       <run>/s1_see/<game>_L<k>_<split>.facts.jsonl  per changing case: the object diff, item by item
       <run>/s1_see/<game>.summary.json              per level and split: the closure numbers
"""
import json
import os

import numpy as np

from gb import artifacts as A
from gb.core.evidence import PALETTE_NAMES, LevelEvidence
from gridtools import GRID, label_components

VERSION = 1                     # bump when what the output means changes (artifacts.py)
KINDS = ("moved", "recoloured", "grew", "shrank", "vanished", "appeared", "reshaped")
UNIT_KINDS = ("moved", "recoloured", "grew", "shrank", "vanished", "appeared")     # all but free-form reshapes
ERASE = ("moved", "grew", "shrank", "reshaped", "vanished")                       # the before-object goes
PAINT = ("moved", "recoloured", "grew", "shrank", "reshaped", "appeared")         # the after-object is drawn
DEFAULTS = {"connectivity": 4, "large_cells": 1024, "compact_items": 8}


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


# ---- one level, one run -------------------------------------------------------------------------------
def see_level(ev, settings=DEFAULTS):
    """Every case of one level's evidence. Returns (arrays per case, records per changing case)."""
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
    records = []
    for k in np.flatnonzero(ev.changed):
        k = int(k)
        r = see_case(ev.before[k], ev.after[k], ev.live, background, settings, objs)
        arr["changed_cells"][k], arr["items"][k], arr["objects"][k] = r["changed"], len(r["items"]), r["n_objects"]
        arr["closure"][k], arr["unit"][k], arr["explained"][k] = r["closure"], r["unit"], r["explained"]
        arr["absorbed"][k] = r["absorbed"]
        for kind in KINDS:
            arr[f"n_{kind}"][k] = sum(it["kind"] == kind for it in r["items"])
        records.append({"k": k, "group": ev.group[k], "changed": r["changed"], "closure": r["closure"],
                        "absorbed": r["absorbed"],
                        "unit": round(r["unit"], 4), "explained": r["explained"],
                        "items": [item_record(it) for it in r["items"]]})
    return arr, records


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
        arr, records = see_level(ev, settings)
        p_npz, p_jsonl = os.path.join(out, name + ".facts.npz"), os.path.join(out, name + ".facts.jsonl")
        np.savez_compressed(p_npz, **arr)
        with open(p_jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summ[name] = summary(ev, arr, settings)
        run.write_header("s1_see", name, VERSION, settings, {"evidence": p_ev}, {"facts": p_npz, "records": p_jsonl},
                         code=__file__)
        s = summ[name]
        log(f"[{name}] {s['changing']} changing cases: closure {s['closure']:.1%}, explained {s['explained']:.1%}, "
            f"whole-object cells {s['unit_cells']:.1%}, with a reshaped object {s['with_reshaped']:.1%}, "
            f"items median {s['items_median']:g} (p90 {s['items_p90']:g})")
    with open(p_sum, "w") as f:
        json.dump(summ, f, indent=1)
    return summ
