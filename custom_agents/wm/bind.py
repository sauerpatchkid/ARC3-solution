"""bind.py - fitting a rule's PARAMS to a level by search, with no LLM call
(docs/plans/rulebook-v2.md section 4.3), and running a rule with a binding.

SEARCH. For one rule on one level's evidence, in ONE child process:
  1. The level's bindings: every combination of the slots' values, up to 4,096.
     With more, bind()'s guess fixes slots (largest first) until the rest fits;
     a rule that cannot be cut down that way fails. A slot the code never reads
     takes its first value. bind()'s whole guess is tried first, then the
     binding carried from the previous level (`prefer`), then the rest in order.
  2. Successive halving: every binding on 10 cases, the survivors on 50, the
     survivors of that on all. A binding survives only while it is exact on
     every claimed cell of every case it applies to. The 10 and the 50 are the
     changing cases of the rule's own kind of action first (ACTION3, or clicks),
     in a fixed shuffled order, because other cases cannot tell bindings apart.
  3. Every binding that is exact on all cases is kept, in this order: most
     applicable cases; then bind()'s guess; then most slots unchanged from the
     previous level; then search order. A binding that applies to nothing is
     exact too, and comes last. The first few are run again to get their
     per-case results. If no exact binding applies to anything, the two that
     predicted the most changed cells before their first mistake are run on
     every case as well, so the rule can be graded and its mistakes shown.

The search works to a time budget. If it runs out, the bindings not yet tried
are counted in `untried` and the result says so; it never reports "exact" for a
binding it did not run on every case.

RUN. run_rule() runs one rule with one given binding on every case of some
evidence: the training binding untouched on the next level (T0), or the fit
split's binding on the score split (T1). Region values are numbers in the layout
of the evidence the binding was made on, so the caller names that evidence.

Per case the result has what wm/check.py's worker reports (applies, correct,
changes, phash, claimed_ok, cells_right, cells_false, claimed_n), so
check.grade() and metrics.claimed_grade() read it unchanged. For a rule that
claims only some cells, `correct` and `phash` are of the board with its
unclaimed cells left as they were.
"""
import ast
import time

import numpy as np

from . import contract as C
from .evidence import GRID, LevelEvidence, screen_hash
from .sandbox import run_in_child

STAGES = (10, 50)            # cases in the first two rounds of the halving; the third is all of them
N_WRONG = 12                 # mistakes kept per fully-run binding (a random sample), to show the LLM
N_EXACT, N_INEXACT = 3, 2    # bindings returned with their per-case results
N_LISTED = 64                # exact bindings listed (binding and applicable cases only)
BUDGET_S = 240.0             # search time per rule and level; the child's hard limit is 2x this


def family_of(group):
    """The kind of action a group is: its button, or 'click' for any clicked colour
    (a rule about clicking blue is a rule about clicking some other colour next level)."""
    return group if group.startswith("ACTION") else "click"


def case_order(ev, family=None, seed=0):
    """Every case of `ev`, the informative ones first: changing cases of the
    rule's family, then its other cases, then everything else; shuffled within
    each (seeded) so the first few are not ten copies of one screen."""
    n = len(ev.actions)
    fam = np.array([family is None or family_of(g) == family for g in ev.group], bool)
    tier = np.where(fam & ev.changed & ~ev.frozen, 0, np.where(fam, 1, 2))
    return np.lexsort((np.random.default_rng(seed).permutation(n), tier))


def unused_slots(src, params):
    """Slots that applies(), predict() and their helpers never name: they cannot
    matter, so they are not searched. (PARAMS itself and bind() name every slot.)"""
    try:
        fns = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name != "bind"]
    except SyntaxError:
        return []
    used = {n.value for fn in fns for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return [name for name in params if name not in used]


def _same(a, b):
    """Two binding values, whichever of them has been through JSON."""
    try:
        a, b = C.plain(a), C.plain(b)
    except TypeError:
        return False
    return type(a) is type(b) and a == b


def candidates(params, dom, guess, prefer, unused):
    """(bindings to try in order, how many the level has, why there are none)."""
    total = C.n_bindings(dom)
    if not total:
        empty = next(n for n, v in dom.items() if not v)
        return [], 0, f"slot {empty!r} has no possible value on this level"
    fixed = {n: dom[n][0] for n in unused}
    left = {n: v for n, v in dom.items() if n not in fixed}
    for name in sorted(left, key=lambda n: -len(left[n])):       # too many: bind() fixes the largest slots
        if C.n_bindings({n: left[n] for n in left if n not in fixed}) <= C.MAX_BINDINGS:
            break
        if name not in guess:
            return [], total, (f"{total:,} bindings is more than {C.MAX_BINDINGS:,}, and bind() does not "
                               f"give a value for {name!r}")
        fixed[name] = guess[name]
    first = []
    for b in (guess, prefer or {}):                              # whole bindings to try before the rest
        b = {n: fixed.get(n, b.get(n)) for n in dom}
        if all(v is not None and any(_same(v, x) for x in dom[n]) for n, v in b.items()):
            first.append({n: C.plain(v) for n, v in b.items()})
    out, seen = [], set()
    for b in first + list(C.all_bindings(dom, fixed)):
        key = C.freeze(b)
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out, total, None


class _Run:
    """One rule with one binding over some cases of one level's evidence."""

    def __init__(self, rule, ev, p):
        self.rule, self.ev, self.p = rule, ev, p
        self.applied = self.missed = self.errors = self.cells = self.done = 0
        self.first_error = None

    def quick(self, cases):
        """Stop at the first claimed-cell miss. True if there was none."""
        ev, live = self.ev, self.ev.live
        for k in cases:
            board, act, api = ev.inputs(k)
            try:
                if self.rule.applies(board, act, api, self.p):
                    pred, claim = self.rule.predict(board, act, api, self.p, live)
                    self.applied += 1
                    after = ev.after[k]
                    if not (pred[claim] == after[claim]).all():
                        self.missed += 1
                        return False
                    self.cells += int((claim & (after != board) & live).sum())
            except Exception as e:
                self.errors += 1
                self.first_error = self.first_error or f"{type(e).__name__}: {e}"
            self.done += 1
        return True

    def full(self, seed=0):
        """Every case, with the per-case arrays (see the module doc)."""
        ev, live = self.ev, self.ev.live
        n = len(ev.actions)
        applies, correct, changes, claimed_ok = (np.zeros(n, bool) for _ in range(4))
        phash = np.zeros(n, np.uint64)
        cells_right, cells_false, claimed_n = (np.zeros(n, np.int32) for _ in range(3))
        ms = np.zeros(n)
        errors, first_error = 0, None
        rng, wrong, n_wrong = np.random.default_rng(seed), [], 0
        for k in range(n):
            board, act, api = ev.inputs(k)
            t0 = time.perf_counter()
            try:
                if self.rule.applies(board, act, api, self.p):
                    pred, claim = self.rule.predict(board, act, api, self.p, live)
                    after = ev.after[k]
                    whole = np.where(claim, pred, board)          # unclaimed cells: left as they were
                    really = (after != board) & live
                    applies[k] = True
                    correct[k] = bool(np.array_equal(whole[live], after[live]))
                    changes[k] = bool((pred != board)[claim].any())
                    phash[k] = screen_hash(whole, live)
                    claimed_ok[k] = bool((pred[claim] == after[claim]).all())
                    cells_right[k] = int((claim & really & (pred == after)).sum())
                    cells_false[k] = int((claim & ~really & (pred != board)).sum())
                    claimed_n[k] = int(claim.sum())
                    if not claimed_ok[k]:                         # reservoir sample of the mistakes
                        n_wrong += 1
                        keep = (k, np.clip(whole, 0, 15).astype(np.uint8), claim.copy())
                        if len(wrong) < N_WRONG:
                            wrong.append(keep)
                        elif (j := int(rng.integers(0, n_wrong))) < N_WRONG:
                            wrong[j] = keep
            except Exception as e:
                errors += 1
                first_error = first_error or f"{type(e).__name__}: {e}"
                if errors > 0.05 * n + 5:                         # systematic, not a rare edge case
                    raise RuntimeError(first_error)
            ms[k] = (time.perf_counter() - t0) * 1000.0
        return {"applies": applies, "correct": correct, "changes": changes, "phash": phash,
                "claimed_ok": claimed_ok, "cells_right": cells_right, "cells_false": cells_false,
                "claimed_n": claimed_n, "ms_mean": float(ms.mean()) if n else 0.0,
                "errors": errors, "first_error": first_error, "wrong": wrong}


def _search_worker(src, ev_path, family, prefer, budget_s, seed, q):
    """Child process: the whole search for one rule on one level (module doc)."""
    try:
        t0 = time.time()
        late = lambda share: time.time() - t0 > share * budget_s
        rule = C.Rule(src)
        ev = LevelEvidence.load(ev_path)
        api = ev.api
        dom = C.domains(rule.params, ev)
        guess, guess_note = rule.guess(api)
        unused = unused_slots(src, rule.params)
        prefer = {n: v for n, v in (prefer or {}).items()       # a region number means nothing on another level
                  if n in rule.params and not isinstance(rule.params[n], C.Region)}
        cands, total, why = candidates(rule.params, dom, guess, prefer, unused)
        out = {"ok": True, "rule": rule.text, "version": rule.version, "bindings": total,
               "guess": guess or None, "guess_note": guess_note, "unused": unused, "halving": [], "untried": 0}
        if why:
            q.put(dict(out, finals=[], exact=0, exact_applying=0, exact_list=[], reason=why))
            return
        order = case_order(ev, family, seed)
        runs = [_Run(rule, ev, rule.values(b, api.layout)) for b in cands]
        alive, done = list(range(len(runs))), 0
        for size in STAGES:
            cases = order[done:size]
            tried = 0
            for i in alive:
                if late(0.4):                                     # the rest of the budget is for the full runs
                    break
                runs[i].quick(cases)
                tried += 1
            out["untried"] += len(alive) - tried
            alive = [i for i in alive[:tried] if not runs[i].missed]
            done = min(size, len(order))
            out["halving"].append({"cases": done, "tried": tried, "exact": len(alive)})
        ran = [r for r in runs if r.done or r.errors]
        if ran and all(r.errors and r.errors >= r.done for r in ran):               # it never ran at all
            raise RuntimeError(next(r.first_error for r in ran))
        exact = []
        queue = sorted(alive, key=lambda i: (-runs[i].applied, i))
        for n, i in enumerate(queue):
            if late(0.8):
                out["untried"] += len(queue) - n
                break
            if runs[i].quick(order[done:]):
                exact.append(i)
        is_guess = lambda i: bool(guess) and all(n in guess and _same(v, guess[n]) for n, v in cands[i].items())
        kept = lambda i: sum(1 for n, v in cands[i].items() if n in prefer and _same(v, prefer[n]))
        exact.sort(key=lambda i: (-runs[i].applied, not is_guess(i), -kept(i), i))
        picked = exact[:N_EXACT]
        if not any(runs[i].applied for i in exact):            # nothing exact says anything: add the best tries
            tried = [i for i in range(len(runs)) if i not in set(exact) and (runs[i].done or runs[i].missed)]
            picked += sorted(tried, key=lambda i: (-runs[i].cells, -runs[i].applied, -runs[i].done, i))[:N_INEXACT] or \
                ([] if picked else [0])
        finals, failure = [], None
        for i in picked:
            if finals and late(1.0):                              # the first is always run; the others if time allows
                break
            try:
                finals.append({"binding": cands[i], "order": i, "exact": i in exact, "guess": is_guess(i),
                               "res": runs[i].full(seed)})
            except RuntimeError as e:
                failure = failure or str(e)
        if not finals:
            raise RuntimeError(failure or "no binding could be run")
        listed = [{"binding": cands[i], "applies": runs[i].applied, "guess": is_guess(i)} for i in exact[:N_LISTED]]
        q.put(dict(out, finals=finals, exact=len(exact), exact_applying=sum(1 for i in exact if runs[i].applied),
                   exact_list=listed, reason="ok"))
    except Exception as e:
        q.put({"ok": False, "stage": "runtime", "reason": f"{type(e).__name__}: {e}"})


def search(src, ev_path, family=None, prefer=None, budget_s=BUDGET_S, seed=0):
    """Fit one rule's PARAMS to one level (module doc). `family`: the rule's
    kind of action (family_of its group). `prefer`: the binding it had on the
    previous level. Returns a dict: ok, stage ('static' | 'runtime' | 'timeout'
    | 'checked') and, when checked:
      bindings    how many the level has;  untried: how many the budget left out
      halving     [{cases, tried, exact}] for the 10- and 50-case rounds
      exact       how many bindings are exact on every case (exact_applying: how
                  many of those apply to at least one case)
      exact_list  the first 64 of them: [{binding, applies (cases), guess}]
      finals      the first 3 of them with their per-case results, and, if none
                  of those applies to anything, the 2 inexact ones that did best:
                  [{binding, order, exact, guess, res}]
      guess       bind()'s answer;  unused: slots the code never reads
    A v1 rule, or a v2 rule without PARAMS, has the one empty binding."""
    ok, why, _, _ = C.check_form(src)
    if not ok:
        return {"ok": False, "stage": "static", "reason": why}
    res = run_in_child(_search_worker, (src, ev_path, family, prefer, budget_s, seed), 2 * budget_s + 60)
    if res["ok"]:
        res["stage"] = "checked"
    return res


def _run_worker(src, ev_path, binding, layout_path, seed, q):
    try:
        rule = C.Rule(src)
        ev = LevelEvidence.load(ev_path)
        layout = (ev if layout_path in (None, ev_path) else LevelEvidence.load(layout_path)).api.layout
        res = _Run(rule, ev, rule.values(binding or {}, layout)).full(seed)
        q.put(dict(res, ok=True, rule=rule.text))
    except Exception as e:
        q.put({"ok": False, "stage": "runtime", "reason": f"{type(e).__name__}: {e}"})


def run_rule(src, ev_path, binding=None, layout_path=None, timeout_s=600, seed=0):
    """One rule with one binding on every case of `ev_path`. `layout_path`: the
    evidence whose layout the binding's region numbers refer to (default: the
    same evidence). Returns what check.check_rule returns."""
    ok, why, _, _ = C.check_form(src)
    if not ok:
        return {"ok": False, "stage": "static", "reason": why}
    res = run_in_child(_run_worker, (src, ev_path, binding, layout_path, seed), timeout_s)
    if res["ok"]:
        res["stage"] = "checked"
    return res


def choose(found):
    """Which binding a level uses, from a search result: (final or None, flag).
    It is the first exact binding in the search's order (module doc, step 3).
    Flags are the failure taxonomy's (pre-registration section 9), plus one:
      None                 one exact binding that applies to something
      'binding ambiguous'  several such bindings; the first is used
      'binding untested'   nothing the rule says could be checked on this level
                           (it applies to no case). Kept only if bind() gave the
                           binding, or if the rule has no PARAMS to bind
      'binding failed'     no binding is exact, or it applies to no case and
                           there is no bind() guess to go on"""
    best = next((f for f in found.get("finals", []) if f["exact"]), None)
    if best is None:
        return None, "binding failed"
    if not best["res"]["applies"].any():
        return (best, "binding untested") if best["guess"] or not best["binding"] else (None, "binding failed")
    return best, ("binding ambiguous" if found["exact_applying"] > 1 else None)
