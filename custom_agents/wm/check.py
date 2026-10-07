"""check.py - is a rule right, how far can it be trusted, and what does the whole
rule book cover (docs/plans/llm-rulebook.md sections 3.2 and 6.3).

THE CHECK. A rule is run, in a child process, on every key of a level's
evidence. Where applies() is True, predict() must reproduce the key's recorded
(majority) outcome EXACTLY on every non-decoration cell. Counts are in recorded
MOVES: a key seen 40 times counts 40.

TRUST, in rising order:
  admitted        applies to >= 20 moves, exactly right on >= 95% of them, and
                  GAIN > 0 (changes predicted exactly, minus changes predicted
                  where nothing happened). Stage A's rule, unchanged. An
                  admitted rule is a hypothesis: fit for repair and diagnostics.
  plan-eligible   admitted, right on EVERY key it applies to, and none of those
                  keys has conflicting outcomes. Only these may drive a plan: a
                  20-step plan on a 95% rule fails about 64% of the time.
  known no-op     predicts "nothing changes" on every key it applies to, covers
                  >= 20 moves, is right on every one, and none of those keys has
                  conflicting outcomes (a screen where the same action sometimes
                  DID change something is not a no-op). Used to prune search.

THE BOOK. Per action group: its best plan-eligible changing rule (highest gain)
and its best known no-op (most moves). For a key the book returns the single
prediction its applicable rules agree on, or UNKNOWN when none applies or they
disagree. UNKNOWN is never "nothing changes": an uncovered move is not evidence
that the move does nothing.

COVERAGE (gate G1) = the share of the level's distinct CHANGING keys the book
predicts exactly; UNKNOWN and wrong both count as misses.
"""
import collections
import time

import numpy as np
import xxhash

from .evidence import GRID, LevelEvidence
from .sandbox import compile_functions, run_in_child, validate

ADMIT = {"min_moves": 20, "min_accuracy": 0.95, "max_mean_ms": 5.0}
N_WRONG = 12                 # mistakes kept per checked rule (a random sample), to show the LLM
UNKNOWN, EXACT, WRONG = 0, 1, 2


def _rule_worker(src, ev_path, seed, q):
    """Child process: run one rule over every key of one level's evidence."""
    try:
        ns = compile_functions(src)
        ev = LevelEvidence.load(ev_path)
        n = len(ev.actions)
        applies, correct, changes = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        phash = np.zeros(n, np.uint64)
        ms = np.zeros(n)
        errors, first_error = 0, None
        rng, wrong, n_wrong = np.random.default_rng(seed), [], 0
        live = ev.live
        for k in range(n):
            board, act, api = ev.inputs(k)
            t0 = time.perf_counter()
            try:
                if ns["applies"](board, act, api):
                    pred = np.asarray(ns["predict"](board, act, api))
                    if pred.shape != (GRID, GRID):
                        raise ValueError(f"predict must return a 64x64 board, got shape {pred.shape}")
                    applies[k] = True
                    correct[k] = ev.exact(k, pred)
                    changes[k] = bool((pred[live] != board[live]).any())
                    phash[k] = xxhash.xxh64(np.where(live, pred, 255).astype(np.uint8).tobytes()).intdigest()
                    if not correct[k]:                    # reservoir sample of the mistakes
                        n_wrong += 1
                        keep = (k, np.clip(pred, 0, 15).astype(np.uint8))
                        if len(wrong) < N_WRONG:
                            wrong.append(keep)
                        elif (j := int(rng.integers(0, n_wrong))) < N_WRONG:
                            wrong[j] = keep
            except Exception as e:
                errors += 1
                first_error = first_error or f"{type(e).__name__}: {e}"
                if errors > 0.05 * n + 5:                  # systematic, not a rare edge case
                    raise RuntimeError(first_error)
            ms[k] = (time.perf_counter() - t0) * 1000.0
        q.put({"ok": True, "rule": str(ns.get("RULE", ""))[:300], "applies": applies, "correct": correct,
               "changes": changes, "phash": phash, "ms_mean": float(ms.mean()) if n else 0.0,
               "errors": errors, "first_error": first_error, "wrong": wrong})
    except Exception as e:
        q.put({"ok": False, "stage": "runtime", "reason": f"{type(e).__name__}: {e}"})


def check_rule(src, ev_path, timeout_s=600, seed=0):
    """Validate and run one rule on one level's saved evidence. Returns a dict:
    ok, stage ('static' | 'runtime' | 'timeout' | 'checked'), reason, and when
    checked the per-key arrays applies / correct / changes / phash."""
    ok, why = validate(src)
    if not ok:
        return {"ok": False, "stage": "static", "reason": why}
    res = run_in_child(_rule_worker, (src, ev_path, seed), timeout_s)
    if res["ok"]:
        res["stage"] = "checked"
    return res


def grade(res, ev, group=None):
    """Trust level of a checked rule on evidence `ev` (see the module doc).
    Adds the counts the feedback prompt and the report use."""
    ap, co = res["applies"], res["correct"]
    w, ag, ch = ev.count, ev.agree, ev.changed
    moves = int(w[ap].sum())
    right = int(ag[ap & co].sum())
    acc = right / moves if moves else 0.0
    changes_right = int(ag[ap & co & ch].sum())
    false_changes = int(w[ap & ~co & ~ch].sum())
    gain = changes_right - false_changes
    conflicts = int((ev.n_variants[ap] > 1).sum())
    exact_all = bool(co[ap].all()) if ap.any() else False
    noop = bool(ap.any()) and not bool(res["changes"][ap].any())
    admitted = (moves >= ADMIT["min_moves"] and acc >= ADMIT["min_accuracy"] and gain > 0
                and res["ms_mean"] <= ADMIT["max_mean_ms"])
    out = {"moves": moves, "keys": int(ap.sum()), "accuracy": round(acc, 4), "gain": gain,
           "changes_right": changes_right, "false_changes": false_changes,
           "conflict_keys": conflicts, "exact_all": exact_all,
           "admitted": admitted,
           "plan_eligible": admitted and exact_all and conflicts == 0,
           "known_noop": noop and moves >= ADMIT["min_moves"] and exact_all and conflicts == 0,
           "reason": "ok"}
    if not admitted and not out["known_noop"]:
        out["reason"] = (
            f"applies to only {moves} recorded moves (needs {ADMIT['min_moves']})" if moves < ADMIT["min_moves"] else
            f"right on {acc:.0%} of the {moves} moves it applies to (needs {ADMIT['min_accuracy']:.0%})"
            if acc < ADMIT["min_accuracy"] else
            "predicts no change exactly (a 'does nothing' rule adds nothing)" if gain <= 0 else
            f"too slow: {res['ms_mean']:.1f} ms per move")
    if group is not None:
        in_g = np.array([g == group for g in ev.group])
        out.update(group_moves=int(w[in_g].sum()), in_group_moves=int(w[ap & in_g].sum()),
                   missed=np.flatnonzero(in_g & ~ap & ch & ~ev.frozen).tolist())
    return out


def pick_book(cands):
    """The rule book from graded candidates (dicts with 'group', 'grade', ...):
    per group the best plan-eligible changing rule and the best known no-op.
    Also returns the Stage A style book (best ADMITTED rule per group), for
    comparison only."""
    by_group = collections.defaultdict(list)
    for c in cands:
        if c.get("stage") == "checked":
            by_group[c["group"]].append(c)
    book, admitted_book = [], []
    for g, cs in sorted(by_group.items()):
        pe = [c for c in cs if c["grade"]["plan_eligible"]]
        if pe:
            book.append(max(pe, key=lambda c: (c["grade"]["gain"], -c["order"])))
        no = [c for c in cs if c["grade"]["known_noop"]]
        if no:
            book.append(max(no, key=lambda c: (c["grade"]["moves"], -c["order"])))
        ad = [c for c in cs if c["grade"]["admitted"]]
        if ad:
            admitted_book.append(max(ad, key=lambda c: (c["grade"]["gain"], -c["order"])))
    return book, admitted_book


def book_status(results, n):
    """Per key: EXACT / WRONG / UNKNOWN for a book, from its rules' check
    results on one evidence set. UNKNOWN when no rule applies or two disagree."""
    status = np.full(n, UNKNOWN, np.int8)
    if not results:
        return status
    ap = np.stack([r["applies"] for r in results])
    ph = np.stack([r["phash"] for r in results])
    co = np.stack([r["correct"] for r in results])
    for k in np.flatnonzero(ap.any(axis=0)):
        who = np.flatnonzero(ap[:, k])
        if len(set(ph[who, k].tolist())) == 1:
            status[k] = EXACT if co[who[0], k] else WRONG
    return status


def coverage(status, ev):
    """Gate G1's number and its companions, over DISTINCT keys."""
    ch = ev.changed
    n_ch = int(ch.sum())

    def share(mask, of):
        return round(float(mask.sum()) / of, 4) if of else 0.0
    return {"changing_keys": n_ch,
            "coverage": share((status == EXACT) & ch, n_ch),
            "wrong": share((status == WRONG) & ch, n_ch),
            "unknown": share((status == UNKNOWN) & ch, n_ch),
            "unchanged_keys": int((~ch).sum()),
            "unchanged_known": share((status == EXACT) & ~ch, int((~ch).sum())),
            "unchanged_wrong": share((status == WRONG) & ~ch, int((~ch).sum()))}


# --------------------------------------------------------------------------------
# Baselines for transfer (no LLM), copied from legacy/llm_track/rule_referee.py
# --------------------------------------------------------------------------------
PATCH, WINDOW = 4, 8         # memory baseline: 9x9 key, change within 8 cells


def _patch(board, y, x):
    padded = np.pad(board, PATCH, constant_values=255)
    return padded[y:y + 2 * PATCH + 1, x:x + 2 * PATCH + 1].tobytes()


def _delta(before, after, y, x, live):
    y0, y1, x0, x1 = max(0, y - WINDOW), min(GRID, y + WINDOW + 1), max(0, x - WINDOW), min(GRID, x + WINDOW + 1)
    ch = np.argwhere((before[y0:y1, x0:x1] != after[y0:y1, x0:x1]) & live[y0:y1, x0:x1])
    return tuple((int(r + y0 - y), int(c + x0 - x), int(after[r + y0, c + x0])) for r, c in ch)


def baseline_memory(train, test):
    """Per test key: is "repeat what the same screen + action did in training,
    or else what the same 9x9 click neighbourhood did" exactly right?"""
    live = train.live & test.live
    exact = {}
    local = collections.defaultdict(collections.Counter)
    for i in range(len(train.actions)):
        b, a, act = train.before[i], train.after[i], int(train.actions[i])
        exact[(np.where(live, b, 255).tobytes(), act)] = i
        if act >= 5:
            y, x = divmod(act - 5, GRID)
            local[_patch(b, y, x)][_delta(b, a, y, x, live)] += int(train.count[i])
    ok = np.zeros(len(test.actions), bool)
    for k in range(len(test.actions)):
        b, a, act = test.before[k], test.after[k], int(test.actions[k])
        key = (np.where(live, b, 255).tobytes(), act)
        if key in exact:
            pred = train.after[exact[key]]
        elif act >= 5 and _patch(b, *divmod(act - 5, GRID)) in local:
            y, x = divmod(act - 5, GRID)
            pred = b.copy()
            for dy, dx, c in local[_patch(b, y, x)].most_common(1)[0][0]:
                if 0 <= y + dy < GRID and 0 <= x + dx < GRID:
                    pred[y + dy, x + dx] = c
        else:
            pred = b
        ok[k] = np.array_equal(pred[test.live], a[test.live])
    return ok


def transfer(status, train, test):
    """How a book frozen on `train` does on `test` (the next level), before any
    repair. One-step exact rates over the test level's distinct keys: the book
    with UNKNOWN read as "nothing changes" (so it is comparable with the
    baselines, as in Stage A), "nothing changes" alone, and memory."""
    ch = test.changed
    book_ok = (status == EXACT) | ((status == UNKNOWN) & ~ch)
    mem = baseline_memory(train, test)
    n = len(ch)
    r = lambda v: round(float(v.mean()), 4) if n else 0.0
    return {"keys": n, "book": r(book_ok), "nothing": r(~ch), "memory": r(mem),
            "beats_both": bool(n and book_ok.mean() > (~ch).mean() and book_ok.mean() > mem.mean()),
            **{"test_" + k: v for k, v in coverage(status, test).items()}}
