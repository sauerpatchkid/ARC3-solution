"""h3_templates.py - H3 TEMPLATES: rules learned on dev games, tried on a new game by re-binding.

The problem it fixes: every new game starts from zero rules, even when a dev game had
the same mechanism (plan section 5.9).

Version 0 is the plan's SMOKE TEST (section 7.1: "a throwaway catalog from G1's trusted
rules, tried on the two seen games only"), not the registered catalog G3 uses:

  CATALOG  every trusted rule of a finished offline run's books (the composed book's
           members), on the dev games, that has PARAMS and no lint warning (a rule
           with level facts in its code cannot carry), one per distinct code.
           Each keeps its source game, group and family: `click`, or `button` for
           any ACTIONn (a rule about ACTION1 on one game may be about ACTION3 on
           another; the search finds out).
  TRY      on a new game's training level, for each changing action group, every
           template of the same family is re-bound by bind.search (the same search
           S4 uses between levels), and graded by v2's trust (book.trust). A template
           is ADOPTED for a group only if it is trusted there.
  BOOK     the adopted templates composed as CP1 composed books (s3_check), so the
           cell coverage they reach together can be set beside the LLM's book.

Nothing here touches a held-out game: `try_game` refuses any game outside the seen
tier (tiers.SEEN), until G3's catalog is built and registered (plan 5.9, steps 1-2).

Reads  <book run>/s3_check/<game>_L<k>.book.json, s2_write/<game>/candidates.jsonl
       <run>/evidence/<game>_L<k>_train.npz        (S0 on the new game)
Writes <run>/h3_templates/catalog.jsonl, <game>.json
"""
import glob
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import tiers
from gb.artifacts import plain
from gb.core import bind as B
from gb.core import book as K
from gb.core.evidence import LevelEvidence
from gb.s3_check import compose_book, member

VERSION = 0
WORKERS = 8


def family(group):
    return "click" if group.startswith("click") else "button"


def build_catalog(book_dir, games):
    """Module doc, CATALOG: [{tid, game, group, family, code, binding, source_id}]."""
    out, seen = [], set()
    for game in games:
        books = glob.glob(os.path.join(book_dir, "s3_check", f"{game}_L*.book.json"))
        if not books:
            continue
        with open(books[0]) as f:
            members = set(json.load(f)["members"])
        with open(os.path.join(book_dir, "s2_write", game, "candidates.jsonl")) as f:
            for c in map(json.loads, f):
                if c["id"] not in members or c.get("lint") or not c.get("binding"):
                    continue
                h = hashlib.sha256(c["code"].encode()).hexdigest()[:16]
                if h in seen:
                    continue
                seen.add(h)
                out.append({"tid": len(out), "game": game, "group": c["group"], "family": family(c["group"]),
                            "code": c["code"], "binding": c["binding"], "source_id": c["id"], "code_hash": h})
    return out


def try_one(t, group, p_train, ev):
    """One template re-bound on one group's level: {tid, group, adopted, flag, binding, trust}."""
    found = B.search(t["code"], p_train, family=B.family_of(group), prefer=t["binding"])
    if found.get("stage") != "checked":
        return {"tid": t["tid"], "group": group, "adopted": False, "flag": "binding failed",
                "why": found.get("reason")}
    best, flag = B.choose(found)
    if best is None:
        return {"tid": t["tid"], "group": group, "adopted": False, "flag": flag, "untried": found["untried"]}
    tr = K.trust([(best["res"], ev)])
    return {"tid": t["tid"], "group": group, "adopted": bool(tr["trusted"]), "flag": flag,
            "binding": best["binding"], "untried": found["untried"],
            "trust": {k: v for k, v in tr.items() if k != "levels"}}


def try_game(run, game, catalog, log=print):
    """Module doc, TRY and BOOK, for one seen game whose evidence S0 put in `run`."""
    if game not in tiers.SEEN:
        raise ValueError(f"{game} is not a seen game: H3's smoke test may only touch {', '.join(tiers.SEEN)}")
    trains = sorted(glob.glob(os.path.join(run.dir, "evidence", f"{game}_L*_train.npz")))
    if len(trains) != 1:
        raise FileNotFoundError(f"{run.name}: expected one training split for {game}, found {len(trains)}")
    p_train = trains[0]
    ev = LevelEvidence.load(p_train)
    groups = [g for g in ev.groups if ev.changed[ev.keys_of(g)].any()]     # groups whose moves change the board
    jobs = [(t, g) for g in groups for t in catalog if t["family"] == family(g)]
    with ThreadPoolExecutor(WORKERS) as ex:
        tried = list(ex.map(lambda tg: try_one(tg[0], tg[1], p_train, ev), jobs))
    adopted = [r for r in tried if r["adopted"]]
    by_tid = {t["tid"]: t for t in catalog}
    cands = [dict(id=i, code=by_tid[r["tid"]]["code"], binding=r["binding"], group=r["group"], lint=[], order=i,
                  trust=r["trust"]) for i, r in enumerate(adopted)]
    offer = sorted(cands, key=lambda c: (groups.index(c["group"]), -c["trust"]["gain_cells"], c["order"]))
    numbers = None
    if offer:
        book, comp = compose_book(offer, p_train)
        numbers = dict(K.numbers(comp, ev), members=[adopted[c["id"]]["tid"] for c in book])
    out = os.path.join(run.dir, "h3_templates")
    os.makedirs(out, exist_ok=True)
    summary = {"game": game, "level": ev.level + 1, "groups": groups, "catalog": len(catalog), "tried": len(tried),
               "adopted": [dict(r, source=f"{by_tid[r['tid']]['game']} {by_tid[r['tid']]['group']}") for r in adopted],
               "untried_bindings": sum(r.get("untried", 0) for r in tried), "book": numbers,
               "results": tried, "version": VERSION}
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump(plain(summary), f, indent=1)
    log(f"[{game}] H3 smoke: {len(groups)} changing groups, {len(tried)} template tries, {len(adopted)} adopted"
        + (f"; their book: {len(numbers['members'])} rules, cell coverage {numbers['cell_coverage']:.1%}"
           if numbers else ""))
    return summary


def write_catalog(run, catalog):
    out = os.path.join(run.dir, "h3_templates")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "catalog.jsonl"), "w") as f:
        for t in catalog:
            f.write(json.dumps(plain(t)) + "\n")
