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
"""
import numpy as np

from .check import ADMIT, EXACT, UNKNOWN, baseline_memory, coverage

MIN_CELLS = 20               # changed cells an admitted rule must predict right in total


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
