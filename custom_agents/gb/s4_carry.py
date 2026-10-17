"""s4_carry.py - S4 CARRY: the book's rules re-bound on the next level, and how they do there.

The problem it fixes: a rule's PARAMS hold one level's facts (its colours, its
regions), so a book frozen on one level says little about the next until each
rule is re-bound there (plan section 5.4). Until now only CP1's import scored
transfer, with CP1's own re-bindings, so nothing new could be carried.

Version 0 is CP1's procedure (legacy/rulebook/tools/wm_tier0a_v2.py, rebind and
score_book), nothing more: each rule of the training level's book is fitted to
the transfer level's FIT split by bind.search (its training binding preferred),
bind.choose picks the binding, and the rule is in T1 if that binding is exact
there, the rule is trusted over both levels, and the lint finds no level fact
in its code. T0 (the book untouched) and T1 are scored on the SCORE split, per
move, against "nothing" and "memory", with the gate's interval
(metrics.gate_interval). Role hints, the online wait and repair calls (plan
5.4) come later. The search runs to a time budget (bind.BUDGET_S), so a slow
machine could stop it early; `untried` says when it did.

Reads  <run>/evidence/<game>_L<k>_train.npz, <game>_L<j>_{fit,score}.npz
       <run>/s2_write/<game>/candidates.jsonl
       <run>/s3_check/<game>_L<k>.book.json
Writes <run>/s4_carry/<game>_L<j>.carry.json      each rule's re-binding, T0 and T1
       <run>/s4_carry/<game>_L<j>.t1_status.npz   per case of the score split: T1's status
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gb.artifacts import plain
from gb.core import bind as B
from gb.core import book as K
from gb.core.evidence import LevelEvidence
from gb.core.metrics import gate_interval, memory_hash, moves_right, nothing_hash
from gb.s3_check import member, regrade, status_arrays

VERSION = 1                     # bump when what the output means changes (artifacts.py)
WORKERS = 8                     # rules re-bound at the same time (each in its own child process)


def rebind(c, res_train, train, fit, p_fit):
    """One rule re-bound on the fit split (CP1's rebind): {in_t1, flag, why, binding, ...}."""
    found = B.search(c["code"], p_fit, family=B.family_of(c["group"]), prefer=c["binding"])
    if found["stage"] != "checked":
        return {"in_t1": False, "flag": "binding failed", "why": found.get("reason"), "binding": None}
    best, flag = B.choose(found)
    out = {"in_t1": False, "flag": flag, "binding": None, "untried": found["untried"]}
    if best is None:
        return dict(out, why="no binding is exact on the fit split")
    t = K.trust([(res_train, train), (best["res"], fit)])
    out.update(binding=best["binding"], trust={k: v for k, v in t.items() if k != "levels"})
    if c["lint"]:
        return dict(out, flag="level-bound rule", why="the lint finds level facts in its code")
    return dict(out, in_t1=t["trusted"], why=None if t["trusted"] else f"not trusted across both levels: {t['reason']}")


def score_book(members, p_score, score, base):
    """A book on the score split: its numbers, per-move rates and interval (CP1's score_book)."""
    comp = K.compose(members, p_score)
    if comp["stage"] != "checked":
        raise RuntimeError(f"the book could not be composed: {comp.get('reason')}")
    nums = K.numbers(comp, score)
    gate = gate_interval(score, dict(base, book=moves_right(score, comp["hash"])))
    return dict(nums, per_move=gate, rules_in=len(members),
                clashing_rules=[i for i, c in enumerate(comp["clashes"]) if c]), comp


def baselines(train, fit, score):
    """Per recorded move of the score split: is "nothing" right, is "memory" right."""
    return {"nothing": moves_right(score, nothing_hash(score)),
            "memory": moves_right(score, memory_hash([train, fit], score))}


def run_game(run, game, log=print, force=False):
    """S4 for one game in `run` (module doc). Returns {t0, t1}, or None if the
    game has no transfer level or the files are current."""
    ev_dir = os.path.join(run.dir, "evidence")
    books = sorted(glob.glob(os.path.join(run.stage_dir("s3_check"), f"{game}_L*.book.json")))
    if not books:
        raise FileNotFoundError(f"{run.name}: no book for {game}; run S3 first")
    with open(books[0]) as f:
        book = json.load(f)
    k = book["level"]
    later = sorted(int(os.path.basename(p)[len(game) + 2:].split("_")[0])
                   for p in glob.glob(os.path.join(ev_dir, f"{game}_L*_fit.npz")))
    j = next((x for x in later if x > k), None)
    p_train = os.path.join(ev_dir, f"{game}_L{k}_train.npz")
    p_fit, p_score = (os.path.join(ev_dir, f"{game}_L{j}_{s}.npz") for s in ("fit", "score"))
    if j is None or not os.path.exists(p_score):
        log(f"[{game}] S4: no transfer level with a score split; nothing to carry")
        return None
    p_cands = os.path.join(run.stage_dir("s2_write"), game, "candidates.jsonl")
    inputs = {"train": p_train, "fit": p_fit, "score": p_score, "candidates": p_cands, "book": books[0]}
    if not force and run.is_current("s4_carry", game, VERSION, {}, inputs):
        log(f"[{game}] S4 is current; skipped")
        return None
    with open(p_cands) as f:
        by_id = {c["id"]: c for c in map(json.loads, f)}
    rules = [by_id[i] for i in book["members"]]
    train, fit, score = (LevelEvidence.load(p) for p in (p_train, p_fit, p_score))
    with ThreadPoolExecutor(WORKERS) as ex:
        res_train = list(ex.map(lambda c: regrade(c, p_train, train)[0], rules))
        carried = list(ex.map(lambda cr: rebind(cr[0], cr[1], train, fit, p_fit), zip(rules, res_train)))
    base = baselines(train, fit, score)
    t0, _ = score_book([member(c, p_train) for c in rules], p_score, score, base)
    kept = [(c, t) for c, t in zip(rules, carried) if t["in_t1"]]
    t1, comp1 = score_book([member(c, p_fit, t["binding"]) for c, t in kept], p_score, score, base)
    out = run.stage_dir("s4_carry")
    p_carry, p_t1 = os.path.join(out, f"{game}_L{j}.carry.json"), os.path.join(out, f"{game}_L{j}.t1_status.npz")
    with open(p_carry, "w") as f:
        json.dump(plain({"game": game, "from_level": k, "level": j,
                         "rules": [{"id": c["id"], "group": c["group"], "binding_before": c["binding"], **t}
                                   for c, t in zip(rules, carried)],
                         "t0": t0, "t1": t1, "source": "s4_carry"}), f, indent=1)
    np.savez_compressed(p_t1, **status_arrays(comp1), members=np.array([c["id"] for c, _ in kept]))
    run.write_header("s4_carry", game, VERSION, {}, inputs, {"carry": p_carry, "t1_status": p_t1}, code=__file__)
    untried = sum(t.get("untried", 0) for t in carried)
    log(f"[{game}] S4: level {k} -> {j}: {len(kept)} of {len(rules)} rules in T1; T1 {t1['per_move']['rate']['book']:.1%} "
        f"(nothing {t1['per_move']['rate']['nothing']:.1%}, memory {t1['per_move']['rate']['memory']:.1%}), "
        f"wrong {t1['wrong']:.1%}" + (f"; WARNING: the search left {untried} bindings untried" if untried else ""))
    return {"t0": t0, "t1": t1}
