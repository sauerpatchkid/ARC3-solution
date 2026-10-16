"""book.py - trust across every level seen, and the composed rule book
(docs/plans/rulebook-v2.md sections 4.4 and 4.5; the definitions the CP1 gate
uses are in docs/plans/rulebook-v2-prereg.md section 4.2).

TRUST. A rule is graded on every level it has been seen on, each with that
level's own binding (wm/bind.py):
  admitted   over those levels together: applies to >= 20 moves, >= 95%
             claimed-exact, gain > 0, >= 20 changed cells predicted right, and
             <= 5 ms per move on each level
  trusted    admitted, and on EVERY level 100% claimed-exact with none of the
             cases it applies to having conflicting outcomes. A level it applies
             to nowhere does not count against it.

THE BOOK. Up to 4 rules per action group. Rules are offered in the caller's
order (best first). A group's first rule joins; a later one joins only if, on at
least 10 cases, it explains (claims and gets right) a changed cell that no rule
already in the book claims. For a case, the book's prediction is put together
cell by cell from the rules that apply:
  a cell one rule claims, or several claim with the same value   that value
  a cell rules claim with DIFFERENT values                       UNKNOWN; every
                                                                 rule involved is flagged for repair
  a cell no rule claims                                          UNKNOWN
and the case is
  UNKNOWN   no cell is claimed
  WRONG     some claimed cell is wrong
  EXACT     the whole board is right when UNKNOWN cells are left as they were
  PARTIAL   every claimed cell is right, but some changed cell is UNKNOWN
A rule that claims a cell stays the same is wrong on it if it changed.

All rules run together in ONE child process per level, because composing needs
every rule's claim on the same case. `hash` is the composed board with UNKNOWN
cells left as they were: what the gate's per-move rate compares with each move's
own next screen (wm.metrics.moves_right).
"""
import collections
import time

import numpy as np

from . import contract as C
from .check import ADMIT, EXACT, UNKNOWN, WRONG
from .evidence import LevelEvidence, screen_hash
from .metrics import MIN_CELLS, changed_cells, claimed_grade
from .sandbox import run_in_child

PARTIAL = 3                  # beside wm.check's UNKNOWN, EXACT, WRONG
MAX_PER_GROUP = 4            # rules per action group
MIN_NEW_CASES = 10           # cases on which a later rule must explain cells the book leaves unclaimed
N_OPEN = 4000                # PARTIAL cases whose unexplained cells are sent back (for residual prompts)


# --------------------------------------------------------------------------------
# Trust across levels
# --------------------------------------------------------------------------------
def trust(levels):
    """Every-level trust of one rule. `levels`: [(res, ev)] for each level seen,
    `res` the rule's per-case results there with THAT level's binding."""
    moves = right = cells_right = cells_false = conflicts = 0
    exact_everywhere, ms, grades = True, 0.0, []
    for res, ev in levels:
        ap, ok = res["applies"], res["claimed_ok"]
        grades.append(dict(claimed_grade(res, ev), level=ev.level + 1))
        moves += int(ev.count[ap].sum())
        right += int(ev.agree[ap & ok].sum())
        cells_right += int((ev.agree[ap] * res["cells_right"][ap]).sum())
        cells_false += int((ev.agree[ap] * res["cells_false"][ap]).sum())
        conflicts += int((ev.n_variants[ap] > 1).sum())
        exact_everywhere &= bool(ok[ap].all())
        ms = max(ms, res["ms_mean"])
    exactness = right / moves if moves else 0.0
    gain = cells_right - cells_false
    admitted = (moves >= ADMIT["min_moves"] and exactness >= ADMIT["min_accuracy"] and gain > 0
                and cells_right >= MIN_CELLS and ms <= ADMIT["max_mean_ms"])
    reason = "ok"
    if not admitted:
        reason = (f"applies to only {moves} recorded moves (needs {ADMIT['min_moves']})" if moves < ADMIT["min_moves"] else
                  f"its claimed cells are all right on {exactness:.0%} of the {moves} moves it applies to "
                  f"(needs {ADMIT['min_accuracy']:.0%})" if exactness < ADMIT["min_accuracy"] else
                  "it claims more changes that did not happen than changes that did" if gain <= 0 else
                  f"claim too narrow: it predicts only {cells_right} changed cells right in total (needs {MIN_CELLS})"
                  if cells_right < MIN_CELLS else f"too slow: {ms:.1f} ms per move")
    elif not exact_everywhere:
        reason = "a claimed cell is wrong on some move it applies to"
    elif conflicts:
        reason = f"it applies to {conflicts} cases that have been seen to end in more than one way"
    return {"moves": moves, "claimed_exactness": round(exactness, 4), "cells_right": cells_right,
            "cells_false": cells_false, "gain_cells": gain, "conflict_cases": conflicts,
            "exact_everywhere": exact_everywhere, "admitted": admitted,
            "trusted": admitted and exact_everywhere and conflicts == 0, "reason": reason, "levels": grades}


# --------------------------------------------------------------------------------
# Composing: the child process
# --------------------------------------------------------------------------------
def _book_worker(members, ev_path, select, q):
    """Child process: run every member rule on every case of one level, choose
    the book if `select`, and compose its prediction case by case."""
    try:
        ev = LevelEvidence.load(ev_path)
        live, n = ev.live, len(ev.actions)
        layouts, rules = {None: ev.api.layout, ev_path: ev.api.layout}, []
        for m in members:
            rule = C.Rule(m["src"])
            lp = m.get("layout_path")
            if lp not in layouts:
                layouts[lp] = LevelEvidence.load(lp).api.layout
            rules.append((rule, rule.values(m.get("binding") or {}, layouts[lp])))
        # -- pass 1: what each rule says about each case, kept small ----------------------------
        said = [dict() for _ in rules]         # case -> (claim: flat cells or None for all live, changed cells, their values)
        errors = [0] * len(rules)
        first_error = [None] * len(rules)
        ms = np.zeros(len(rules))
        flat_live = live.ravel()
        for k in range(n):
            board, act, api = ev.inputs(k)
            flat = board.ravel()
            for j, (rule, p) in enumerate(rules):
                t0 = time.perf_counter()
                try:
                    if rule.applies(board, act, api, p):
                        pred, claim = rule.predict(board, act, api, p, live)
                        pf, cf = np.clip(pred, 0, 255).astype(np.uint8).ravel(), claim.ravel()
                        diff = np.flatnonzero(cf & (pf != flat)).astype(np.int16)
                        said[j][k] = (None if claim is live else np.flatnonzero(cf).astype(np.int16), diff, pf[diff])
                except Exception as e:
                    errors[j] += 1
                    first_error[j] = first_error[j] or f"{type(e).__name__}: {e}"
                ms[j] += (time.perf_counter() - t0) * 1000.0

        def claim_mask(j, k):
            c = said[j][k][0]
            if c is None:
                return flat_live
            m = np.zeros(flat_live.size, bool)
            m[c] = True
            return m

        # -- which rules are in the book --------------------------------------------------------
        notes, book, per_group = [None] * len(rules), [], collections.Counter()

        def really(k):                          # the cells of a case that really changed
            return (ev.before[k].ravel() != ev.after[k].ravel()) & flat_live

        for j, m in enumerate(members):
            g = m.get("group")
            if errors[j] > 0.05 * n + 5:
                notes[j] = f"failed to run: {first_error[j]}"
                continue
            if not select or m.get("locked"):
                book.append(j)
                per_group[g] += 1
                continue
            if per_group[g] >= MAX_PER_GROUP:
                notes[j] = f"its group already has {MAX_PER_GROUP} rules"
                continue
            if per_group[g]:                    # a later rule must add something the book does not claim
                adds = 0
                for k in said[j]:
                    after = ev.after[k].ravel()
                    mine = claim_mask(j, k) & really(k)
                    if not mine.any():
                        continue
                    right = np.zeros(flat_live.size, bool)
                    _, diff, vals = said[j][k]
                    right[diff] = vals == after[diff]
                    mine &= right
                    for i in book:
                        if k in said[i] and mine.any():
                            mine &= ~claim_mask(i, k)
                    adds += bool(mine.any())
                if adds < MIN_NEW_CASES:
                    notes[j] = (f"claim too narrow: it explains cells the book leaves unclaimed on {adds} cases "
                                f"(needs {MIN_NEW_CASES})")
                    continue
            book.append(j)
            per_group[g] += 1
        # -- pass 2: compose the book's prediction for every case ----------------------------------
        status = np.full(n, UNKNOWN, np.int8)
        phash = np.zeros(n, np.uint64)
        n_rules, cells_right, cells_wrong, cells_clash, claimed_n = (np.zeros(n, np.int32) for _ in range(5))
        clashes, open_cells = [0] * len(rules), {}
        for k in range(n):
            flat, after = ev.before[k].ravel(), ev.after[k].ravel()
            who = [j for j in book if k in said[j]]
            n_rules[k] = len(who)
            out = flat.copy()
            if who:
                claimed = np.zeros(flat.size, bool)
                clash = np.zeros(flat.size, bool)
                if len(who) == 1:
                    j = who[0]
                    claimed = claim_mask(j, k).copy()
                    out[said[j][k][1]] = said[j][k][2]
                else:
                    masks = {j: claim_mask(j, k) for j in who}
                    value = {j: flat.copy() for j in who}
                    for j in who:
                        value[j][said[j][k][1]] = said[j][k][2]
                        claimed |= masks[j]
                    touched = np.unique(np.concatenate([said[j][k][1] for j in who])) if who else []
                    for c in touched:           # only a cell some rule changes can be claimed two ways
                        vals = {int(value[j][c]) for j in who if masks[j][c]}
                        if len(vals) > 1:
                            clash[c] = True
                            for j in who:
                                if masks[j][c]:
                                    clashes[j] += 1
                        elif vals:
                            out[c] = vals.pop()
                    claimed &= ~clash
                wrong = claimed & (out != after)
                cells_right[k] = int((claimed & really(k) & (out == after)).sum())
                cells_wrong[k] = int(wrong.sum())
                cells_clash[k] = int(clash.sum())
                claimed_n[k] = int(claimed.sum())
                if claimed.any():
                    status[k] = (WRONG if wrong.any() else
                                 EXACT if np.array_equal(out[flat_live], after[flat_live]) else PARTIAL)
                    if status[k] == PARTIAL and len(open_cells) < N_OPEN:
                        open_cells[k] = np.packbits(really(k) & ~claimed)
            phash[k] = screen_hash(out.reshape(live.shape), live)
        q.put({"ok": True, "book": book, "notes": notes, "status": status, "hash": phash, "n_rules": n_rules,
               "cells_right": cells_right, "cells_wrong": cells_wrong, "cells_clash": cells_clash,
               "claimed_n": claimed_n, "clashes": clashes, "errors": errors, "open": open_cells,
               "applies": [len(s) for s in said], "ms_mean": (ms / max(n, 1)).tolist()})
    except Exception as e:
        q.put({"ok": False, "stage": "runtime", "reason": f"{type(e).__name__}: {e}"})


def compose(members, ev_path, select=False, timeout_s=1800):
    """The composed book on one level. `members`: [{src, binding, layout_path,
    group, locked}] in priority order (best first). With `select` the book is
    chosen from them by the joining rule (locked members are in regardless);
    without it every member is in the book. Returns per case `status`, `hash`,
    `n_rules`, `cells_right`, `cells_wrong`, `cells_clash`, `claimed_n`; `book`
    (the members that are in, by position), `notes` (why a member is not),
    `clashes` (per member: cells it claimed against another rule), and `open`
    (see unexplained())."""
    if not members:
        ev = LevelEvidence.load(ev_path)
        n = len(ev.actions)
        zeros = lambda: np.zeros(n, np.int32)
        return {"ok": True, "stage": "checked", "book": [], "notes": [], "status": np.full(n, UNKNOWN, np.int8),
                "hash": np.array([screen_hash(b, ev.live) for b in ev.before], np.uint64), "n_rules": zeros(),
                "cells_right": zeros(), "cells_wrong": zeros(), "cells_clash": zeros(), "claimed_n": zeros(),
                "clashes": [], "errors": [], "open": {}, "applies": [], "ms_mean": []}
    for m in members:
        ok, why, _, _ = C.check_form(m["src"])
        if not ok:
            return {"ok": False, "stage": "static", "reason": why}
    res = run_in_child(_book_worker, (members, ev_path, select), timeout_s)
    if res["ok"]:
        res["stage"] = "checked"
    return res


# --------------------------------------------------------------------------------
# What a composed book scores
# --------------------------------------------------------------------------------
def numbers(res, ev):
    """The three numbers and the safety number for a composed book on one level:
      claimed_exactness  of the moves where the book claims something, the share
                         with every claimed cell right
      cell_coverage      of the really changed cells (distinct cases), the share
                         the book claims and gets right
      case_coverage      v1's strict number: the share of distinct changing cases
                         whose whole board is right
      wrong              the share of distinct changing cases with a wrong
                         claimed cell (the gate's safety number); `partial` and
                         `unknown` are the rest of the changing cases
    """
    st, ch = res["status"], ev.changed
    spoke = st != UNKNOWN
    moves = int(ev.count[spoke].sum())
    right = int(ev.agree[spoke & (st != WRONG)].sum())
    total = int(changed_cells(ev).sum())
    n_ch = int(ch.sum())
    share = lambda mask, of: round(float(mask.sum()) / of, 4) if of else 0.0
    return {"rules": len(res["book"]), "changing_cases": n_ch,
            "claimed_exactness": round(right / moves, 4) if moves else 0.0,
            "cell_coverage": round(int(res["cells_right"].sum()) / total, 4) if total else 0.0,
            "case_coverage": share((st == EXACT) & ch, n_ch),
            "wrong": share((st == WRONG) & ch, n_ch),
            "partial": share((st == PARTIAL) & ch, n_ch),
            "unknown": share((st == UNKNOWN) & ch, n_ch),
            "unchanged_wrong": share((st == WRONG) & ~ch, int((~ch).sum())),
            "clash_cases": int((res["cells_clash"] > 0).sum())}


def unexplained(res, ev, group=None):
    """{case: 64x64 bool} of changed cells the composed book leaves unclaimed, on
    changing cases (of one action group, if given): the leftover cells of PARTIAL
    cases, and every changed cell of a case the book says nothing about. Frozen
    stretches are left out, as in every prompt."""
    out = {}
    for k in np.flatnonzero(ev.changed & ~ev.frozen):
        k = int(k)
        if group is not None and ev.group[k] != group:
            continue
        if res["status"][k] == UNKNOWN:
            out[k] = (ev.before[k] != ev.after[k]) & ev.live
        elif k in res["open"]:
            out[k] = np.unpackbits(res["open"][k])[:ev.live.size].reshape(ev.live.shape).astype(bool)
    return out
