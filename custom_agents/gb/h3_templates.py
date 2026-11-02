"""h3_templates.py - H3 TEMPLATES: rules learned on dev games, tried on a new game by re-binding.

The problem it fixes: every new game starts from zero rules, even when a dev game had
the same mechanism (plan section 5.9).

Version 1 (2 Nov) is the catalog G3 uses and the way a new game tries it. Version 0,
the plan's smoke test (section 7.1: a throwaway catalog tried on the two seen games
only; 18 Oct, 0 of 21 adopted), is kept below exactly as it ran.

  CATALOG  (`build_catalog_v1`, once; then frozen and registered as
           glassbox_templates_v1, `[templates] catalog`). From finished offline runs,
           every candidate trusted on its game's training level, on a dev or seen game
           (tiers.BUILD_FROM), with PARAMS and no lint warning (a rule with level facts
           in its code cannot carry):
             - bind() is kept only if the lint finds no level fact in it. The lint skips
               bind(), so here it reads bind() as if it were applies(); a bind() it
               flags is removed, and the search then has no guess to start from.
             - Duplicates go: first the same code, then the same BEHAVIOUR (what it
               says on a fixed bank of BANK_BOARDS synthetic boards, blobs on a
               background in the style of H1's test fixtures, for the 5 buttons and
               BANK_CLICKS clicks each, under its own binding: whether it applies, which
               cells it claims and their predicted values). Only a template that applies
               to at least BANK_MIN_APPLIES of the bank's moves and raises on none is
               compared by behaviour: a rule that never applies to random blobs says
               nothing there, and every such rule would look the same (the first build,
               2 Nov, dropped tu93's two rules as "duplicates" of tr87's and g50t's that
               way). A template with a Region slot is compared by code only (a region
               number means nothing off its own level). Of two duplicates, the one with
               more cells right is kept.
             - At most `max_templates`, by cells right.
             - Each keeps its family (button or click), KIND (if its code sets one),
               source game, group and candidate, and the levels it was trusted on (the
               training level, and the transfer level if S4 carried it there).
  TRY      (`try_game`) on a new game's training level, for each group S2 would call
           (s2_write.groups_to_call), every template of the group's family is re-bound
           by bind.search with no binding preferred (a source game's colours mean
           nothing on a new game), the group's `search_budget_s` shared among them, and
           graded by v2's trust (book.trust). A template is ADOPTED for the group only
           if it is trusted there.
  USE      In GB+T (G3) S2 starts each group from its adopted templates: a group whose
           template carries skips the first round, and the LLM is asked only for what
           the book does not explain yet (s2_write, `templates`). S3 and S4 treat a
           template like any rule.

Held-out games: `try_game` refuses the untouched tier always, and the A/B-offline tier
unless `g3` is set (only after pre-registration section 6, G3, is registered) and the
catalog file matches its registry entry.

Reads  <book run>/s3_check/<game>_L<k>.book.json, s2_write/<game>/candidates.jsonl,
       s4_carry/<game>_L<j>.carry.json                       (building the catalog)
       <run>/evidence/<game>_L<k>_train.npz                  (S0 on the new game)
Writes the catalog (`[templates] catalog`) and <run>/h3_templates/catalog_report.{json,md}
       <run>/h3_templates/<game>.json                        (a game's tries and adoptions)
"""
import ast
import glob
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import registry
import tiers
from gb.artifacts import ROOT, plain
from gb.core import bind as B
from gb.core import book as K
from gb.core import contract as C
from gb.core import lint as L
from gb.core.evidence import GRID, Act, Api, LevelEvidence
from gb.core.sandbox import run_in_child
from gb.s3_check import compose_book

VERSION = 1
WORKERS = 8
REGISTRY_NAME = "glassbox_templates_v1"
BANK_BOARDS, BANK_CLICKS = 200, 4
BANK_MIN_APPLIES = 20           # bank moves a template must apply to before its behaviour is compared (ADMIT's 20)
DEFAULTS = {"search_budget_s": 240, "max_templates": 60}


def family(group):
    return "click" if group.startswith("click") else "button"


# ---- version 1: the catalog ------------------------------------------------------------------------
def rule_kind(code):
    """The KIND a rule's code sets (KIND = "..."), or None."""
    for node in ast.parse(code).body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KIND" for t in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            return node.value.value
    return None


def checked_bind(code):
    """(code, why bind() was removed or None): bind() kept only if the lint, reading it as
    applies(), finds no level fact in it (module doc, CATALOG)."""
    tree = ast.parse(code)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bind"), None)
    if fn is None:
        return code, None
    warnings = L.lint(ast.get_source_segment(code, fn).replace("def bind(", "def applies(", 1))
    if not warnings:
        return code, None
    lines = code.split("\n")
    start = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
    kept = lines[:start] + lines[fn.end_lineno:]
    why = "; ".join(sorted({f"{w['kind']} ({w['code']})" for w in warnings}))
    return "\n".join(kept), why


def bank(n=BANK_BOARDS, clicks=BANK_CLICKS):
    """The fixed synthetic boards and their moves: [(board, [unified actions])]. Blobs of
    colours 1-15 on black, some touching, one large floor (H1's test fixtures' style,
    every colour so that any binding's colours turn up)."""
    out = []
    for seed in range(n):
        rng = np.random.default_rng(seed)
        b = np.zeros((GRID, GRID), np.uint8)
        b[30:62, 2:40] = 2
        for _ in range(40):
            y, x = rng.integers(0, 60, 2)
            h, w = rng.integers(1, 5, 2)
            b[y:y + h, x:x + w] = rng.integers(1, 16)
        cells = np.argwhere(b != 0)
        picks = cells[rng.choice(len(cells), clicks, replace=False)]
        out.append((b, list(range(5)) + [5 + GRID * int(y) + int(x) for y, x in picks]))
    return out


def _signature_worker(items, q):
    """Child process: each (code, binding)'s answers on the bank, hashed (module doc)."""
    try:
        boards = bank()
        live = np.ones((GRID, GRID), bool)
        out = []
        for code, binding in items:
            h, applies, errors = hashlib.sha256(), 0, 0
            rule = C.Rule(code)
            for board, actions in boards:
                api = Api(board, ~live, 0)
                board.flags.writeable = False
                p = rule.values(binding, api.layout)
                for a in actions:
                    try:
                        if not rule.applies(board, Act(a), api, p):
                            h.update(b"-")
                            continue
                        pred, claim = rule.predict(board, Act(a), api, p, live)
                        h.update(np.packbits(claim).tobytes())
                        h.update(np.asarray(pred, np.uint8)[claim].tobytes())
                        applies += 1
                    except Exception as e:                      # an error is an answer too
                        h.update(type(e).__name__.encode())
                        errors += 1
            out.append((h.hexdigest()[:16], applies, errors))
        q.put({"ok": True, "sigs": out})
    except Exception as e:
        q.put({"ok": False, "reason": f"{type(e).__name__}: {e}"})


def signatures(templates):
    """(signature or None, bank moves it applies to, bank moves it raised on) per template, in one
    sandbox child. None: a Region slot, or too little behaviour on the bank to compare (module doc)."""
    todo = [t for t in templates if not any(isinstance(s, C.Region) for s in C.read_params(t["code"])[1].values())]
    got = run_in_child(_signature_worker, ([(t["code"], t["binding"]) for t in todo],), 1800) if todo else {}
    if todo and not got.get("ok"):
        raise RuntimeError(f"behaviour signatures failed: {got.get('reason')}")
    by_key = dict(zip((t["key"] for t in todo), got.get("sigs", [])))
    out = []
    for t in templates:
        sig, applies, errors = by_key.get(t["key"], (None, None, None))
        out.append((sig if applies is not None and applies >= BANK_MIN_APPLIES and not errors else None,
                    applies, errors))
    return out


def trusted_levels(book_dir, game, level, cid):
    """The levels a book member was trusted on: its training level, and the transfer level
    if S4 carried it there."""
    out = [level]
    for p in glob.glob(os.path.join(book_dir, "s4_carry", f"{game}_L*.carry.json")):
        with open(p) as f:
            carry = json.load(f)
        if any(r["id"] == cid and r.get("in_t1") for r in carry["rules"]):
            out.append(carry["level"])
    return sorted(out)


def build_catalog_v1(book_dirs, max_templates=DEFAULTS["max_templates"], log=print):
    """Module doc, CATALOG. Returns (catalog, report)."""
    considered, report = [], {"sources": [], "bind_removed": [], "duplicate_code": [], "duplicate_behaviour": [],
                              "over_cap": []}
    for book_dir in book_dirs:
        for p_book in sorted(glob.glob(os.path.join(book_dir, "s3_check", "*_L*.book.json"))):
            game = os.path.basename(p_book).split("_L")[0]
            if game not in tiers.BUILD_FROM:
                raise ValueError(f"{game}: templates may come from dev and seen games only")
            with open(p_book) as f:
                level = json.load(f)["level"]
            p_cands = os.path.join(book_dir, "s2_write", game, "candidates.jsonl")
            with open(p_cands) as f:
                cands = [json.loads(line) for line in f]
            n = 0
            for c in cands:
                if not (c.get("stage") == "checked" and (c.get("trust") or {}).get("trusted") and not c.get("lint")
                        and c.get("binding") and not c.get("duplicate")):
                    continue
                code, why = checked_bind(c["code"])
                if why:
                    report["bind_removed"].append({"game": game, "candidate": c["id"], "why": why})
                considered.append({"key": f"{game}/{c['id']}", "game": game, "group": c["group"],
                                   "family": family(c["group"]), "kind": rule_kind(c["code"]), "code": code,
                                   "bind_kept": why is None, "binding": c["binding"], "source_id": c["id"],
                                   "source_run": os.path.basename(book_dir.rstrip("/")),
                                   "rule": c.get("rule", ""), "trusted_on": trusted_levels(book_dir, game, level,
                                                                                         c["id"]),
                                   "cells_right": int(c["trust"]["cells_right"]), "moves": int(c["trust"]["moves"]),
                                   "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16]})
                n += 1
            report["sources"].append({"run": os.path.basename(book_dir.rstrip("/")), "game": game, "level": level,
                                      "candidates": len(cands), "trusted_clean": n})
    considered.sort(key=lambda t: (-t["cells_right"], t["key"]))          # the better of two duplicates first
    kept, by_code, by_sig = [], {}, {}
    for t, (sig, applies, errors) in zip(considered, signatures(considered)):
        t.update(behaviour=sig, bank_applies=applies, bank_errors=errors)
        if t["code_hash"] in by_code:
            report["duplicate_code"].append({"dropped": t["key"], "kept": by_code[t["code_hash"]]})
            continue
        spec = (t["family"], repr(sorted((k, type(v).__name__) for k, v in C.read_params(t["code"])[1].items())))
        if sig is not None and (spec, sig) in by_sig:
            report["duplicate_behaviour"].append({"dropped": t["key"], "kept": by_sig[(spec, sig)]})
            continue
        by_code[t["code_hash"]] = t["key"]
        if sig is not None:
            by_sig[(spec, sig)] = t["key"]
        kept.append(t)
    report["over_cap"] = [t["key"] for t in kept[max_templates:]]
    catalog = [dict(t, tid=i) for i, t in enumerate(kept[:max_templates])]
    report.update(considered=len(considered), templates=len(catalog),
                  behaviour_compared=sum(t["behaviour"] is not None for t in considered),
                  by_family={f: sum(t["family"] == f for t in catalog) for f in ("button", "click")},
                  by_game={g: sum(t["game"] == g for t in catalog) for g in sorted({t["game"] for t in catalog})})
    log(f"H3 catalog: {len(considered)} trusted lint-clean rules with PARAMS considered; "
        f"{len(report['duplicate_code'])} duplicate code, {len(report['duplicate_behaviour'])} duplicate behaviour, "
        f"{len(report['over_cap'])} over the cap; {len(catalog)} templates ({report['by_family']}); "
        f"bind() removed from {len(report['bind_removed'])}")
    return catalog, report


def write_catalog_v1(path, catalog, report_dir, report):
    """The catalog (one template per line) and its report. Refuses to overwrite a registered catalog."""
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    if any(rel in a["files"] for a in registry.load()["artifacts"]):
        raise SystemExit(f"{rel} is registered: a changed catalog needs a new name")
    with open(path, "w") as f:
        for t in catalog:
            f.write(json.dumps(plain({k: v for k, v in t.items() if k != "key"} | {"source": t["key"]}),
                               sort_keys=True) + "\n")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "catalog_report.json"), "w") as f:
        json.dump(plain(report), f, indent=1)
    with open(os.path.join(report_dir, "catalog_report.md"), "w") as f:
        f.write(report_text(catalog, report))
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def report_text(catalog, report):
    lines = [f"# H3 catalog: {len(catalog)} templates", "",
             f"{report['considered']} trusted, lint-clean rules with PARAMS considered; dropped as duplicate code "
             f"{len(report['duplicate_code'])}, as duplicate behaviour {len(report['duplicate_behaviour'])}, over "
             f"the cap {len(report['over_cap'])}; bind() removed from {len(report['bind_removed'])}.", "",
             f"Behaviour was compared for {report['behaviour_compared']} of {report['considered']}: the others apply "
             f"to fewer than {BANK_MIN_APPLIES} of the bank's {BANK_BOARDS * (5 + BANK_CLICKS):,} moves, raise there, "
             "or have a Region slot, so only their code was compared.", "",
             "| Template | Source | Family | Group | Trusted on levels | Moves | Cells right | Bank moves it applies to | "
             "bind() | Rule |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in catalog:
        bank_n = "-" if t["bank_applies"] is None else t["bank_applies"]
        lines.append(f"| {t['tid']} | {t['key']} | {t['family']} | {t['group']} | {t['trusted_on']} | {t['moves']} | "
                     f"{t['cells_right']} | {bank_n} | {'kept' if t['bind_kept'] else 'removed'} | {t['rule'][:90]} |")
    return "\n".join(lines) + "\n"


def load_catalog(path, verify=True):
    """The catalog, after checking (if `verify`) that the file is the registered one."""
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    if verify:
        entry = next((a for a in registry.load()["artifacts"] if a["name"] == REGISTRY_NAME), None)
        if entry is None or rel not in entry["files"]:
            raise SystemExit(f"{rel} is not registered as {REGISTRY_NAME}")
        if registry.verify(REGISTRY_NAME):
            raise SystemExit(f"{rel} differs from its registered hash")
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---- version 1: trying the catalog on a new game ---------------------------------------------------
def try_template(t, group, p_train, ev, budget_s):
    """One template re-bound on one group's training level (module doc, TRY)."""
    found = B.search(t["code"], p_train, family=B.family_of(group), budget_s=budget_s)
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


def try_game(run, game, catalog, groups, settings=DEFAULTS, g3=False, log=print):
    """Module doc, TRY, for one game whose training evidence S0 put in `run`; `groups` are
    the groups S2 would call. Writes <run>/h3_templates/<game>.json."""
    if game in tiers.UNTOUCHED:
        raise ValueError(f"{game} is in the untouched tier: only the confirm may run it")
    if game in tiers.AB_OFFLINE and not g3:
        raise ValueError(f"{game} is an A/B-offline game: templates are tried there only in G3 (pre-registration "
                         "section 6), with g3 set after that section is registered")
    trains = sorted(glob.glob(os.path.join(run.dir, "evidence", f"{game}_L*_train.npz")))
    if len(trains) != 1:
        raise FileNotFoundError(f"{run.name}: expected one training split for {game}, found {len(trains)}")
    p_train = trains[0]
    ev = LevelEvidence.load(p_train)
    tried = []
    for group in groups:
        mine = [t for t in catalog if t["family"] == family(group)]
        budget = settings["search_budget_s"] / max(1, math.ceil(len(mine) / WORKERS))
        with ThreadPoolExecutor(WORKERS) as ex:
            tried += list(ex.map(lambda t: try_template(t, group, p_train, ev, budget), mine))
    adopted = [r for r in tried if r["adopted"]]
    summary = {"game": game, "level": ev.level + 1, "groups": groups, "catalog": len(catalog), "tried": len(tried),
               "adopted": adopted, "untried_bindings": sum(r.get("untried", 0) for r in tried),
               "results": tried, "version": VERSION}
    out = os.path.join(run.dir, "h3_templates")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump(plain(summary), f, indent=1)
    log(f"[{game}] H3: {len(groups)} groups, {len(tried)} template tries, {len(adopted)} adopted"
        + (f" ({', '.join(sorted({r['group'] for r in adopted}))})" if adopted else ""))
    return summary


def adopted_for(run, game, catalog):
    """{group: [{tid, code, binding}]}: what try_game adopted on `game` in `run`, for S2."""
    with open(os.path.join(run.dir, "h3_templates", f"{game}.json")) as f:
        adopted = json.load(f)["adopted"]
    by_tid = {t["tid"]: t for t in catalog}
    out = {}
    for r in adopted:
        out.setdefault(r["group"], []).append({"tid": r["tid"], "code": by_tid[r["tid"]]["code"],
                                               "binding": r["binding"]})
    return out


# ---- version 0: the smoke test (18 Oct), as it ran ---------------------------------------------------
def build_catalog(book_dir, games):
    """v0's catalog: every book member of `book_dir` on `games` with PARAMS and no lint
    warning, one per distinct code: [{tid, game, group, family, code, binding, source_id}]."""
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
    """v0: one template re-bound on one group's level, its source binding preferred."""
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


def smoke_game(run, game, catalog, log=print):
    """v0's smoke test on one SEEN game whose evidence S0 put in `run`: every template of each
    changing group's family tried, the adopted ones composed as CP1 composed books."""
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
               "results": tried, "version": 0}
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump(plain(summary), f, indent=1)
    log(f"[{game}] H3 smoke: {len(groups)} changing groups, {len(tried)} template tries, {len(adopted)} adopted"
        + (f"; their book: {len(numbers['members'])} rules, cell coverage {numbers['cell_coverage']:.1%}"
           if numbers else ""))
    return summary


def write_catalog(run, catalog):
    """v0's throwaway catalog, into the smoke run."""
    out = os.path.join(run.dir, "h3_templates")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "catalog.jsonl"), "w") as f:
        for t in catalog:
            f.write(json.dumps(plain(t)) + "\n")
