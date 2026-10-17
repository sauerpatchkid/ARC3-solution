"""s3_check.py - S3 CHECK, grading: every candidate replayed on its level, and the book composed.

The problem it fixes: a rule is only worth what the replay says, and the book is
only worth what its rules do together (plan section 5.3, steps 1 and 3). Until
now only CP1's import did this, against CP1's own evidence, so a change to the
evidence (S0) could not be graded at all.

Version 0 is CP1's procedure, nothing more: each candidate is run on every case
of the training level with the binding it came with (bind.run_rule) and graded
by v2's trust (book.trust); the book is composed as CP1 composed it: the
trusted rules that are not duplicates or two-level repairs, in the order of the
groups the LLM was called on, then lint-clean first, most gain, earliest; joined
by book.compose's rule (at most 4 per group, a later rule must add cells).
cp1_import.py calls the same functions, so tests/test_gb_regression.py checks
them against CP1's registered numbers. Priorities and the trust tiers (plan
5.3, steps 3 and 4) come later.

Reads  <run>/evidence/<game>_L<k>_train.npz
       <run>/s2_write/<game>/candidates.jsonl, groups.json (the groups called, in order)
Writes <run>/s3_check/<game>/rules.jsonl        each candidate's grade on the level
       <run>/s3_check/<game>/rule_cases.npz     per rule and case: applies, claimed cells right, cells right
       <run>/s3_check/<game>_L<k>.book.json     the book: members, the rules offered, notes, numbers
       <run>/s3_check/<game>_L<k>.status.npz    per case: the book's status
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

VERSION = 1                     # bump when what the output means changes (artifacts.py)
WORKERS = 8                     # candidates graded at the same time (each in its own child process)


def member(c, layout_path, binding=None, **kw):
    """One rule as book.compose takes it."""
    return dict(src=c["code"], binding=c["binding"] if binding is None else binding,
                layout_path=layout_path, group=c["group"], **kw)


def regrade(c, p_train, train):
    """One candidate on its training level, with its own binding: (per-case results, trust), or (None, reason)."""
    res = B.run_rule(c["code"], p_train, binding=c.get("binding") or {})
    if res.get("stage") != "checked":
        return None, res.get("reason")
    return res, K.trust([(res, train)])


def offered(cands, trust_of, groups):
    """The rules offered to the book, best first (CP1's order). `trust_of(c)`:
    a candidate's trust; `groups`: the groups called, in order."""
    order = {g: i for i, g in enumerate(groups)}
    ok = [c for c in cands if c.get("origin") != "two-level" and not c.get("duplicate")
          and (t := trust_of(c)) and t["trusted"]]
    return sorted(ok, key=lambda c: (order.get(c["group"], len(order)), bool(c["lint"]),
                                     -trust_of(c)["gain_cells"], c["order"]))


def compose_book(cands, p_train):
    """(the book's rules, book.compose's result) for the rules offered, in order."""
    comp = K.compose([member(c, p_train) for c in cands], p_train, select=True)
    if comp["stage"] != "checked":
        raise RuntimeError(f"the book could not be composed: {comp.get('reason')}")
    return [cands[i] for i in comp["book"]], comp


def status_arrays(comp):
    return {k: np.asarray(comp[k]) for k in ("status", "hash", "n_rules", "cells_right", "cells_wrong",
                                             "cells_clash", "claimed_n")}


def run_game(run, game, log=print, force=False):
    """S3's grading for one game in `run` (module doc). Returns the book's numbers."""
    trains = sorted(glob.glob(os.path.join(run.dir, "evidence", f"{game}_L*_train.npz")))
    if len(trains) != 1:
        raise FileNotFoundError(f"{run.name}: expected one training split for {game}, found {len(trains)}")
    p_train = trains[0]
    k = int(os.path.basename(p_train)[len(game) + 2:].split("_")[0])
    s2 = os.path.join(run.stage_dir("s2_write"), game)
    p_cands, p_groups = os.path.join(s2, "candidates.jsonl"), os.path.join(s2, "groups.json")
    inputs = {"evidence": p_train, "candidates": p_cands, "groups": p_groups}
    if not force and run.is_current("s3_check", game, VERSION, {}, inputs):
        log(f"[{game}] S3's grading is current; skipped")
        return None
    with open(p_cands) as f:
        cands = [json.loads(line) for line in f if line.strip()]
    with open(p_groups) as f:
        groups = json.load(f)
    train = LevelEvidence.load(p_train)
    gradable = [c for c in cands if c.get("stage") == "checked"]
    with ThreadPoolExecutor(WORKERS) as ex:
        graded = list(ex.map(lambda c: regrade(c, p_train, train), gradable))
    trust = {c["id"]: t for c, (r, t) in zip(gradable, graded) if r is not None}
    rules = [{"id": c["id"], "group": c["group"], "origin": c.get("origin"), "rule": c.get("rule"),
              "binding": c.get("binding"), "lint": c.get("lint"), "duplicate": c.get("duplicate"),
              "v2_admitted": bool(trust.get(c["id"]) and trust[c["id"]]["admitted"]),
              "v2_trusted": bool(trust.get(c["id"]) and trust[c["id"]]["trusted"]),
              "trust": trust.get(c["id"]), "source": c.get("source")} for c in gradable]
    ran = [(c, r) for c, (r, _) in zip(gradable, graded) if r is not None]
    n = len(train.actions)
    out = os.path.join(run.stage_dir("s3_check"), game)
    os.makedirs(out, exist_ok=True)
    p_rules, p_cases = os.path.join(out, "rules.jsonl"), os.path.join(out, "rule_cases.npz")
    with open(p_rules, "w") as f:
        for r in rules:
            f.write(json.dumps(plain(r)) + "\n")
    np.savez_compressed(p_cases, ids=np.array([c["id"] for c, _ in ran], np.int32).reshape(-1),
                        applies=np.array([r["applies"] for _, r in ran], bool).reshape(-1, n),
                        claimed_ok=np.array([r["claimed_ok"] for _, r in ran], bool).reshape(-1, n),
                        cells_right=np.array([r["cells_right"] for _, r in ran], np.int32).reshape(-1, n))
    offer = offered(gradable, lambda c: trust.get(c["id"]), groups)
    book, comp = compose_book(offer, p_train)
    numbers = K.numbers(comp, train)
    p_book = os.path.join(run.stage_dir("s3_check"), f"{game}_L{k}.book.json")
    with open(p_book, "w") as f:
        json.dump(plain({"game": game, "level": k, "members": [c["id"] for c in book],
                         "offered": [c["id"] for c in offer], "groups_called": groups,
                         "notes": {str(c["id"]): t for c, t in zip(offer, comp["notes"]) if t},
                         "numbers": numbers, "source": "s3_check"}), f, indent=1)
    p_status = os.path.join(run.stage_dir("s3_check"), f"{game}_L{k}.status.npz")
    np.savez_compressed(p_status, **status_arrays(comp), members=np.array([c["id"] for c in book]))
    run.write_header("s3_check", game, VERSION, {}, inputs,
                     {"rules": p_rules, "rule_cases": p_cases, "book": p_book, "status": p_status}, code=__file__)
    log(f"[{game}] S3: {len(gradable)} candidates graded, {sum(r['v2_admitted'] for r in rules)} admitted, "
        f"{sum(r['v2_trusted'] for r in rules)} trusted; book {len(book)} rules, cell coverage "
        f"{numbers['cell_coverage']:.1%}, case coverage {numbers['case_coverage']:.1%}")
    return numbers
