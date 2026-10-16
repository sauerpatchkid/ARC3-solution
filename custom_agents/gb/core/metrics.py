"""metrics.py - Rulebook v2's measuring tools (docs/plans/rulebook-v2.md section 3.3).

v1 graded a rule on whole boards: one wrong cell anywhere and the case counted as
wrong, so a rule that explained most of a change counted for nothing. v2 lets a
rule CLAIM the cells it vouches for (wm/check.py reports the per-key results) and
grades it on those. Three numbers are always reported together:

  claimed-cell exactness   of the moves a rule applies to, the share where every
                           claimed cell is right
  cell coverage            of all the cells that really changed on a level, the
                           share some rule claimed and predicted right
  case coverage            v1's strict number: the share of distinct changing
                           cases predicted exactly, whole board (wm.check.coverage)

TRUST, v2 (section 4.5), on ONE level - the every-level version is CP1's:
  admitted   applies to >= 20 moves, >= 95% claimed-exact, gain > 0, >= 20
             changed cells predicted right in total, <= 5 ms per move
  trusted    admitted, 100% claimed-exact, and no conflicting cases
`gain` is in cells: changed cells claimed and right, minus cells claimed as
changing that did not change. The 20-cell floor blocks a rule that claims
almost nothing.

TRANSFER is named three ways (section 3.3): T0 the untouched book, T1 after
automatic re-binding (no LLM), T2 after one repair call. Only T0 exists before
CP1 builds re-binding; transfer_t0() below is v1's transfer() on the SCORE split
of the transfer level, with the memory baseline also given the fit split.

Counts are in recorded moves (a case seen 40 times counts 40) wherever v1's
were, and over distinct cases where v1's coverage was.

THE GATE'S INTERVAL (docs/plans/rulebook-v2-prereg.md section 4.4, settled 13 Oct
2026 before any CP1 run). Rates here are PER MOVE: a move is right for an arm if
the arm's prediction for its case matches that move's OWN recorded next screen
on every non-decoration cell (so a case seen to end two ways cannot be right on
all its moves). The unit resampled is the ATTEMPT (a stretch of recorded moves
between game overs within the score split; its first, partial one counts), or a
block of 50 consecutive moves when the split has fewer than 20 attempts. 2,000
resamples, seed 0; each draws as many units as the split has, with replacement,
and pools their moves. The interval of each difference (book - "nothing", book -
"memory") is the 2.5th to 97.5th percentile of its resampled values, every arm
scored on the same resample. Fewer than 5 units: no interval, and the game does
not count as beating the baselines.
"""
import numpy as np

from .check import ADMIT, EXACT, UNKNOWN, baseline_memory, coverage, memory_boards
from .evidence import screen_hash

MIN_CELLS = 20               # changed cells an admitted rule must predict right in total
MIN_ATTEMPTS, BLOCK, MIN_UNITS, RESAMPLES = 20, 50, 5, 2000      # the gate's interval (pre-registration 4.4)


def claimed_grade(res, ev):
    """v2 grade of one checked rule on one level's evidence."""
    ap, ok = res["applies"], res["claimed_ok"]
    w, ag = ev.count, ev.agree
    moves = int(w[ap].sum())
    right = int(ag[ap & ok].sum())
    exactness = right / moves if moves else 0.0
    cells_right = int((ag[ap] * res["cells_right"][ap]).sum())
    cells_false = int((ag[ap] * res["cells_false"][ap]).sum())
    gain = cells_right - cells_false
    conflicts = int((ev.n_variants[ap] > 1).sum())
    all_ok = bool(ok[ap].all()) if ap.any() else False
    admitted = (moves >= ADMIT["min_moves"] and exactness >= ADMIT["min_accuracy"] and gain > 0
                and cells_right >= MIN_CELLS and res["ms_mean"] <= ADMIT["max_mean_ms"])
    return {"moves": moves, "keys": int(ap.sum()), "claimed_exactness": round(exactness, 4),
            "cells_right": cells_right, "cells_false": cells_false, "gain_cells": gain,
            "mean_claimed_cells": round(float(res["claimed_n"][ap].mean()), 1) if ap.any() else 0.0,
            "conflict_keys": conflicts, "claimed_exact_all": all_ok,
            "admitted_v2": admitted, "trusted_v2": admitted and all_ok and conflicts == 0}


def changed_cells(ev):
    """Per case: how many non-decoration cells really changed."""
    return ((ev.before != ev.after) & ev.live).reshape(len(ev.actions), -1).sum(axis=1)


def cell_coverage(results, ev):
    """Share of the level's really-changed cells (over distinct cases) that the
    rules claim and predict right.

    How several rules combine here is v1's: where the rules that apply to a case
    all predict the same board, their (shared) result counts; where they
    disagree, the case is UNKNOWN and counts nothing. Composing rules with
    DIFFERENT claims on the same case is CP1 (section 4.5) and replaces this."""
    total = int(changed_cells(ev).sum())
    if not results or not total:
        return 0.0
    ap = np.stack([r["applies"] for r in results])
    ph = np.stack([r["phash"] for r in results])
    cr = np.stack([r["cells_right"] for r in results])
    got = 0
    for k in np.flatnonzero(ap.any(axis=0)):
        who = np.flatnonzero(ap[:, k])
        if len(set(ph[who, k].tolist())) == 1:
            got += int(cr[who[0], k])
    return round(got / total, 4)


def three_numbers(results, status, ev):
    """The three numbers reported everywhere, for a set of checked rules on one
    level: claimed-cell exactness (pooled over the rules), cell coverage, case
    coverage. `status` is wm.check.book_status(results, n)."""
    moves = sum(int(ev.count[r["applies"]].sum()) for r in results)
    right = sum(int(ev.agree[r["applies"] & r["claimed_ok"]].sum()) for r in results)
    return {"claimed_exactness": round(right / moves, 4) if moves else 0.0,
            "cell_coverage": cell_coverage(results, ev),
            "case_coverage": coverage(status, ev)["coverage"]}


def transfer_t0(status, train, fit, score):
    """T0: a book frozen on `train`, untouched, on the SCORE split of the next
    level. One-step exact rates over the score split's distinct cases: the book
    with UNKNOWN read as "nothing changes", "nothing changes" alone, and memory
    (which may also use the fit split). `status` is the book's status on `score`."""
    ch = score.changed
    book_ok = (status == EXACT) | ((status == UNKNOWN) & ~ch)
    mem = baseline_memory([train, fit] if fit is not None else [train], score)
    n = len(ch)
    r = lambda v: round(float(v.mean()), 4) if n else 0.0
    return {"cases": n, "book": r(book_ok), "nothing": r(~ch), "memory": r(mem),
            "beats_both": bool(n and book_ok.mean() > (~ch).mean() and book_ok.mean() > mem.mean()),
            "wrong_on_changing": coverage(status, score)["wrong"],
            "per_case": {"book": book_ok, "nothing": ~ch, "memory": mem}}


# --------------------------------------------------------------------------------
# Per-move rates and the gate's interval (see the module doc)
# --------------------------------------------------------------------------------
def nothing_hash(ev):
    """Per case: the hash of "nothing changes" (the board before, decorations blanked)."""
    return np.array([screen_hash(b, ev.live) for b in ev.before], np.uint64)


def memory_hash(train, test):
    """Per test case: the hash of what the memory baseline predicts."""
    return np.array([screen_hash(p, test.live) for p in memory_boards(train, test)], np.uint64)


def v1_book_hash(results, ev):
    """Per case: the hash of what a v1-style book predicts, from its rules'
    check results. Where the rules that apply agree it is their prediction;
    where none applies or they disagree (UNKNOWN) it is "nothing changes"."""
    out = nothing_hash(ev)
    if results:
        ap = np.stack([r["applies"] for r in results])
        ph = np.stack([r["phash"] for r in results])
        for k in np.flatnonzero(ap.any(axis=0)):
            who = np.flatnonzero(ap[:, k])
            if len(set(ph[who, k].tolist())) == 1:
                out[k] = ph[who[0], k]
    return out


def moves_right(ev, pred_hash):
    """Per recorded move of `ev`: does the prediction for its case match the
    move's own next screen? A move whose case the check-set cap dropped has no
    prediction and is never right (resample_units leaves it out of the rates)."""
    key = ev.moves["key"]
    return (key >= 0) & (np.asarray(pred_hash, np.uint64)[np.maximum(key, 0)] == ev.moves["out"])


def resample_units(ev):
    """(unit per move, what the unit is, attempts in the split): the attempt
    each move belongs to, or its block of 50 moves if there are under 20 attempts."""
    if not ev.moves or "attempt" not in ev.moves:
        raise ValueError("this evidence has no move order: rebuild it from the recording (LevelEvidence.split)")
    att = np.unique(ev.moves["attempt"], return_inverse=True)[1]
    n_att = int(att.max()) + 1 if len(att) else 0
    if n_att >= MIN_ATTEMPTS:
        return att, "attempts", n_att
    return np.arange(len(att)) // BLOCK, f"blocks of {BLOCK} moves", n_att


def gate_interval(ev, right, arm="book", baselines=("nothing", "memory"), resamples=RESAMPLES, seed=0):
    """The gate's numbers for one game's score split. `right` maps each arm to
    a bool per recorded move (moves_right). Returns the per-move rates and, for
    each baseline, the difference arm - baseline with its 95% interval;
    `beats_both` is the gate's primary criterion for this game."""
    unit, kind, n_att = resample_units(ev)
    scored = ev.moves["key"] >= 0
    n_units = int(unit.max()) + 1 if len(unit) else 0
    per_unit = lambda v: np.bincount(unit[scored], np.asarray(v, float)[scored], minlength=n_units)
    size = per_unit(np.ones(len(unit)))
    hits = {a: per_unit(right[a]) for a in (arm, *baselines)}
    total = float(size.sum())
    out = {"moves": int(total), "moves_without_a_case": int((~scored).sum()), "attempts": n_att,
           "unit": kind, "units": n_units,
           "rate": {a: round(float(h.sum()) / total, 4) if total else 0.0 for a, h in hits.items()}, "diff": {}}
    draw = None
    if n_units >= MIN_UNITS:
        draw = np.random.default_rng(seed).integers(0, n_units, size=(resamples, n_units))
        pooled = size[draw].sum(axis=1)
    for b in baselines:
        d = {"point": round(float(hits[arm].sum() - hits[b].sum()) / total, 4) if total else 0.0,
             "lo": None, "hi": None, "above_zero": False}
        if draw is not None:
            lo, hi = np.percentile((hits[arm][draw].sum(axis=1) - hits[b][draw].sum(axis=1)) / pooled, [2.5, 97.5])
            d.update(lo=round(float(lo), 4), hi=round(float(hi), 4), above_zero=bool(lo > 0))
        out["diff"][b] = d
    out["beats_both"] = all(d["above_zero"] for d in out["diff"].values())
    return out
