"""cp1_import.py - Rulebook v2's CP1, re-run with gb/core and written as Glassbox artifacts.

The problem it fixes: gb/core is a COPY of Rulebook v2's code, and Glassbox's
first gate (G1) is judged against CP1's numbers. If the copy drifted, every
later comparison with CP1 would be wrong without anyone noticing. Importing CP1
also gives S6 SHOW real data to draw before any Glassbox stage exists.

What is re-run, on the CPU, with no LLM and no binding search:
  1. every candidate rule, graded again on its training level with the binding
     CP1 chose for it (gb/core: bind.run_rule, book.trust);
  2. the book, composed again from CP1's trusted rules in CP1's order;
  3. T0, T1, T1 without untested rules, and T2 on the transfer level's score
     split, with CP1's bindings, beside "nothing" and "memory", with the gate's
     interval.
Each recomputed number is put beside the one CP1 registered. They must be equal.

Not re-run: the LLM (CP1's answers are taken from its file) and the binding
search (its time budget makes it depend on the machine's speed). The book logic
follows legacy/rulebook/tools/wm_tier0a_v2.py (finish_game), re-written here
because legacy/ is never imported.

Reads results/rulebook/v2/tier0a_v2/ (registered: read only, never written).
Writes results/glassbox/<run>/:
  evidence/<game>_L<k>_{train,fit,score}.npz   copies of CP1's evidence
  s2_write/<game>/candidates.jsonl             CP1's candidates, as recorded
  s3_check/<game>/rules.jsonl                  every candidate re-graded on its training level
  s3_check/<game>_L<k>.book.json               the book: members, notes, numbers
  s3_check/<game>_L<k>.status.npz              per case of the training level: the book's status
  s4_carry/<game>_L<k+1>.carry.json            each book rule's re-binding (CP1's) and T0/T1/T2
  s4_carry/<game>_L<k+1>.t1_status.npz         per case of the score split: T1's status
  cp1_check/<game>.json                        every recomputed number beside CP1's, and whether they match
"""
import collections
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gb import artifacts as A
from gb.core import bind as B
from gb.core import book as K
from gb.core.evidence import LevelEvidence
from gb.core.metrics import gate_interval, memory_hash, moves_right, nothing_hash

CP1_DIR = os.path.join(A.ROOT, "results", "rulebook", "v2", "tier0a_v2")
VERSION = 1                     # of what the imported files mean (artifacts.py)
WORKERS = 8                     # candidates graded at the same time (each in its own child process)


# ---- reading CP1 --------------------------------------------------------------------------
def cp1_games(src=CP1_DIR):
    return sorted(f[:-5] for f in os.listdir(src) if f.endswith(".json") and f[:-5].isalnum())


def load_cp1(game, src=CP1_DIR):
    with open(os.path.join(src, f"{game}.json")) as f:
        return json.load(f)


def is_trusted(c):
    return c.get("stage") == "checked" and c["trust"]["trusted"]


def member(c, layout_path, binding=None, **kw):
    """One rule as book.compose takes it (CP1's as_member)."""
    return dict(src=c["code"], binding=c["binding"] if binding is None else binding,
                layout_path=layout_path, group=c["group"], **kw)


# ---- comparing ------------------------------------------------------------------------------
def plain(x):
    """Numbers as JSON holds them, so a recomputed dict and a stored one compare equal."""
    return json.loads(json.dumps(x, default=lambda o: o.item() if isinstance(o, np.generic) else list(o)))


class Check:
    """Recomputed numbers beside CP1's registered ones."""

    def __init__(self, game):
        self.game, self.items = game, []

    def same(self, what, ours, cp1):
        ours, cp1 = plain(ours), plain(cp1)
        self.items.append({"what": what, "match": ours == cp1, "ours": ours, "cp1": cp1})

    @property
    def ok(self):
        return all(i["match"] for i in self.items)

    def summary(self):
        bad = [i["what"] for i in self.items if not i["match"]]
        return {"game": self.game, "checked": len(self.items), "match": len(self.items) - len(bad),
                "mismatches": bad, "ok": not bad}


# ---- one game -------------------------------------------------------------------------------
def regrade(c, p_train, train):
    """One candidate on its training level, with the binding CP1 graded it with."""
    res = B.run_rule(c["code"], p_train, binding=c.get("binding") or {})
    if res.get("stage") != "checked":
        return None, res.get("reason")
    return res, K.trust([(res, train)])


def score_book(members, p_score, score, base):
    """A book on the score split: its numbers, per-move rates and interval (CP1's score_book)."""
    comp = K.compose(members, p_score)
    if comp["stage"] != "checked":
        raise RuntimeError(f"the book could not be composed: {comp.get('reason')}")
    nums = K.numbers(comp, score)
    gate = gate_interval(score, dict(base, book=moves_right(score, comp["hash"])))
    return dict(nums, per_move=gate, rules_in=len(members),
                clashing_rules=[i for i, c in enumerate(comp["clashes"]) if c]), comp


def _status_arrays(comp):
    return {k: np.asarray(comp[k]) for k in ("status", "hash", "n_rules", "cells_right", "cells_wrong",
                                             "cells_clash", "claimed_n")}


def import_game(game, run, src=CP1_DIR, workers=WORKERS, log=print):
    """Re-run one game of CP1 into `run` (an artifacts.Run). Returns its Check."""
    res = load_cp1(game, src)
    check = Check(game)
    k_train = res["train"]["level"]
    k_next = res.get("transfer_level")
    cands = res["candidates"]

    # -- evidence: copies of CP1's ----------------------------------------------------------------
    ev_dir = os.path.join(run.dir, "evidence")
    os.makedirs(ev_dir, exist_ok=True)
    paths = {"train": os.path.join(ev_dir, f"{game}_L{k_train}_train.npz")}
    if k_next:
        paths.update(fit=os.path.join(ev_dir, f"{game}_L{k_next}_fit.npz"),
                     score=os.path.join(ev_dir, f"{game}_L{k_next}_score.npz"))
    for split, dst in paths.items():
        if os.path.exists(os.path.join(src, f"{game}_{split}.npz")):
            shutil.copyfile(os.path.join(src, f"{game}_{split}.npz"), dst)
    paths = {s: p for s, p in paths.items() if os.path.exists(p)}
    train = LevelEvidence.load(paths["train"])
    inputs = {"cp1_result": os.path.join(src, f"{game}.json"), **{f"cp1_{s}": p for s, p in paths.items()}}
    settings = {"imported_from": os.path.relpath(src, A.ROOT)}

    # -- S2: CP1's candidates, as recorded ------------------------------------------------------------
    s2 = os.path.join(run.stage_dir("s2_write"), game)
    os.makedirs(s2, exist_ok=True)
    p_cands = os.path.join(s2, "candidates.jsonl")
    with open(p_cands, "w") as f:
        for c in cands:
            f.write(json.dumps(dict(c, source="cp1")) + "\n")
    run.write_header("s2_write", game, VERSION, settings, inputs, {"candidates": p_cands})

    # -- S3: every candidate re-graded on the training level ---------------------------------------
    gradable = [c for c in cands if c.get("stage") == "checked" and "trust" in c]
    with ThreadPoolExecutor(workers) as ex:
        graded = list(ex.map(lambda c: regrade(c, paths["train"], train), gradable))
    rules = []
    for c, (r, t) in zip(gradable, graded):
        check.same(f"rule {c['id']} trust on the training level", t, c["trust"])
        rules.append({"id": c["id"], "group": c["group"], "origin": c.get("origin"), "rule": c.get("rule"),
                      "binding": c.get("binding"), "lint": c.get("lint"), "duplicate": c.get("duplicate"),
                      "v2_admitted": bool(t and t["admitted"]), "v2_trusted": bool(t and t["trusted"]),
                      "trust": t, "in_book": c.get("in_book", False), "t1": c.get("t1"), "source": "cp1"})
    log(f"[{game}] re-graded {len(gradable)} candidates: "
        f"{sum(i['match'] for i in check.items)} match CP1")

    # -- S3: the book on the training level, composed again (CP1's finish_game) ------------------
    order = {g["group"]: i for i, g in enumerate(res["groups"])}
    built = [c for c in cands if c.get("origin") != "two-level"]          # repairs came after the book
    trusted = sorted((c for c in built if is_trusted(c) and not c.get("duplicate")),
                     key=lambda c: (order[c["group"]], bool(c["lint"]), -c["trust"]["gain_cells"], c["order"]))
    comp = K.compose([member(c, paths["train"]) for c in trusted], paths["train"], select=True)
    if comp["stage"] != "checked":
        raise RuntimeError(f"{game}: the book could not be composed: {comp.get('reason')}")
    book = [trusted[i] for i in comp["book"]]
    check.same("book members (candidate ids)", [c["id"] for c in book], res["book"])
    book_train = K.numbers(comp, train)
    check.same("book on the training level", book_train, res["book_train"])
    s3 = run.stage_dir("s3_check")
    os.makedirs(os.path.join(s3, game), exist_ok=True)
    p_rules = os.path.join(s3, game, "rules.jsonl")
    with open(p_rules, "w") as f:
        for r in rules:
            f.write(json.dumps(plain(r)) + "\n")
    p_book = os.path.join(s3, f"{game}_L{k_train}.book.json")
    with open(p_book, "w") as f:
        json.dump(plain({"game": game, "level": k_train, "members": [c["id"] for c in book],
                         "offered": [c["id"] for c in trusted],
                         "notes": {str(c["id"]): n for c, n in zip(trusted, comp["notes"]) if n},
                         "numbers": book_train, "source": "cp1"}), f, indent=1)
    p_status = os.path.join(s3, f"{game}_L{k_train}.status.npz")
    np.savez_compressed(p_status, **_status_arrays(comp), members=np.array([c["id"] for c in book]))
    run.write_header("s3_check", game, VERSION, settings, inputs,
                     {"rules": p_rules, "book": p_book, "status": p_status})

    # -- S4: transfer, scored again with CP1's bindings -------------------------------------------
    if res.get("transfer") and "score" in paths:
        fit, score = LevelEvidence.load(paths["fit"]), LevelEvidence.load(paths["score"])
        base = {"nothing": moves_right(score, nothing_hash(score)),
                "memory": moves_right(score, memory_hash([train, fit], score))}
        tr = res["transfer"]
        t0, _ = score_book([member(c, paths["train"]) for c in book], paths["score"], score, base)
        kept = [c for c in book if c["t1"]["in_t1"]]
        t1, comp1 = score_book([member(c, paths["fit"], c["t1"]["binding"]) for c in kept], paths["score"], score, base)
        untested = [c for c in kept if c["t1"]["flag"] == "binding untested"]
        t1u, _ = score_book([member(c, paths["fit"], c["t1"]["binding"]) for c in kept if c not in untested],
                            paths["score"], score, base)
        good = [c for c in cands if c.get("origin") == "two-level" and (c.get("t1") or {}).get("in_t1")]
        sel = K.compose([member(c, paths["train"], locked=True) for c in kept] +
                        [member(c, paths["train"]) for c in good], paths["train"], select=True)
        t2_rules = (kept + good) if sel["stage"] != "checked" else [(kept + good)[i] for i in sel["book"]]
        t2, _ = score_book([member(c, paths["fit"], c["t1"]["binding"]) for c in t2_rules], paths["score"], score, base)
        drop = ("flags", "repair_calls", "repaired")
        for name, ours in (("t0", t0), ("t1", t1), ("t1_without_untested", t1u), ("t2", t2)):
            check.same(f"{name} on the transfer level", ours, {k: v for k, v in tr[name].items() if k not in drop})
        s4 = run.stage_dir("s4_carry")
        p_carry = os.path.join(s4, f"{game}_L{k_next}.carry.json")
        with open(p_carry, "w") as f:
            json.dump(plain({"game": game, "from_level": k_train, "level": k_next,
                             "rules": [{"id": c["id"], "group": c["group"], "binding_before": c["binding"],
                                        **{k: c["t1"].get(k) for k in ("in_t1", "flag", "why", "binding", "text")}}
                                       for c in book],
                             "t0": t0, "t1": t1, "t1_without_untested": t1u, "t2": t2,
                             "t2_rules": [c["id"] for c in t2_rules], "source": "cp1"}), f, indent=1)
        p_t1 = os.path.join(s4, f"{game}_L{k_next}.t1_status.npz")
        np.savez_compressed(p_t1, **_status_arrays(comp1), members=np.array([c["id"] for c in kept]))
        run.write_header("s4_carry", game, VERSION, settings, inputs, {"carry": p_carry, "t1_status": p_t1})
        log(f"[{game}] transfer level {k_next}: T1 {t1['per_move']['rate']}, wrong {t1['wrong']:.1%}")

    out = os.path.join(run.dir, "cp1_check")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump({"summary": check.summary(), "items": check.items}, f, indent=1)
    s = check.summary()
    log(f"[{game}] {'MATCH' if s['ok'] else 'MISMATCH'}: {s['match']} of {s['checked']} numbers equal CP1's"
        + (f"; differ: {', '.join(s['mismatches'][:5])}" if s["mismatches"] else ""))
    return check
