#!/usr/bin/env python3
"""wm_tier0a_v2.py - Rulebook v2, CP1: Tier 0a v2, the offline test of rules that
carry to the next level (docs/plans/rulebook-v2.md section 4). The gate is
PRE-REGISTERED in docs/plans/rulebook-v2-prereg.md section 4; readings settled
while building, before any answer was generated, are in its section 10.

Held equal to Tier 0a: the 8 dev games, seed 0 of the same recorded runs, the
same training level (the first with >= 500 moves) and transfer level (the next
with >= 50 distinct cases), the same model, thinking on, 20,480-token answers
with one code-only retry and one repair attempt, at most 12 action groups.

For each game:
  1. RULES, on the training level. For each action group whose moves ever change
     the board: 4 first-round candidates, then up to 8 more calls, stopping the
     search at the first rule that is trusted and has no level facts in its
     code. Those calls refine one rule at a time, chosen by CP0's bandit
     (tools/wm_refine.py) on claimed-cell exactness. Once a group has such a
     rule, what is left of its 8 calls goes on residual prompts while at least
     10 of its cases still have changed cells no rule claims. A group whose
     moves never change the board gets no calls: no v2 rule for it can be
     admitted, and a move no rule covers is read as "nothing changes" anyway.
     Every candidate's PARAMS are fitted to the level by search (wm/bind.py).
  2. THE BOOK, on the training level: the trusted rules, up to 4 per group,
     composed cell by cell (wm/book.py).
  3. TRANSFER, on the next level. Its first 300 moves are the FIT split, the
     rest the SCORE split; nothing is fitted to the score split.
       T0  the book untouched, with the training level's bindings.
       T1  each rule's PARAMS fitted again to the fit split, no LLM. A rule is
           left out if the lint marks it level-bound, if no binding is exact on
           the fit split, or if it is not trusted on both levels. A rule that
           applies to nothing in the fit split stays, flagged, only with bind()'s
           guess or with no PARAMS at all.
       T2  T1 plus one two-level repair call for each group that lost a rule.
     Each is scored on the score split per recorded move, beside "nothing
     changes" and memory (which also gets the fit split), with the gate's
     interval (wm/metrics.py).

THE GATE (pre-registration 4.3). Primary: T1 beats both baselines with the 95%
interval of each difference above zero, on >= 3 of 8 games. Safety: T1 is wrong
on <= 5% of the transfer level's changing cases, on every game. PASS = both.

NOT BUILT, by the plan's cut lines (section 4.11): the simplification pass and
api.t (move-by-move grading). The report says so.

  experiments/rulebook/llm_run.sh start tier0a2           # server + this, detached
  uv run python tools/wm_tier0a_v2.py --evidence-only      # no LLM: evidence, baselines, intervals
  uv run python tools/wm_tier0a_v2.py --from-cache         # re-score a finished run, no server
Writes <out>/{<game>.json, <game>_{train,fit,score}.npz, report.md, llm_cache.jsonl}.
"""
import argparse
import collections
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # legacy/rulebook
ROOT = os.path.dirname(os.path.dirname(TRACK))                        # the repo
sys.path.insert(0, TRACK)                                             # the frozen wm package
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
from manifest import read_manifest  # noqa: E402
from wm import bind as B  # noqa: E402
from wm import book as K  # noqa: E402
from wm import contract as C  # noqa: E402
from wm import registry, tiers  # noqa: E402
from wm import rules as R  # noqa: E402
from wm import rules2 as R2  # noqa: E402
from wm.check import ADMIT, grade  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.lint import lint  # noqa: E402
from wm.llm import ChatClient  # noqa: E402
from wm.metrics import gate_interval, memory_hash, moves_right, nothing_hash  # noqa: E402

MANIFEST = "results/confirm_upgrade/20260927_221915/manifest.tsv"
V1 = os.path.join(ROOT, "results", "rulebook", "tier0a")
V1_ROW = os.path.join(ROOT, "results", "rulebook", "v2", "v1_under_v2_cp1")
K_FIRST, CALLS, BANDIT_C, MAX_GROUPS = 4, 8, 20.0, 12
MIN_TRAIN_MOVES, MIN_TRANSFER_KEYS, FIT_MOVES = 500, 50, 300
GROUP_WORKERS = 4                # action groups worked on at the same time
STOP_NEEDS_CLEAN_LINT = True     # a group's search stops at a trusted rule only if the lint passes it
GATE_GAMES, GATE_WRONG, COVERAGE_TARGET = 3, 0.05, 0.80
BROKEN = ("static", "runtime", "timeout", "unbound")       # worth one repair attempt


def rng_for(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def level_moves(corpus):
    """{level: recorded moves} for one run."""
    import glob
    c = collections.Counter()
    for p in sorted(glob.glob(os.path.join(corpus, "shard_*.npz"))):
        with np.load(p) as z:
            for lv, n in zip(*np.unique(z["levels"], return_counts=True)):
                c[int(lv)] += int(n)
    return c


def pick_levels(game, corpus):
    """Tier 0a's levels: (training evidence, transfer level or None, moves per level)."""
    counts = level_moves(corpus)
    train_level = next((lv for lv in sorted(counts) if counts[lv] >= MIN_TRAIN_MOVES), min(counts))
    train = LevelEvidence.from_corpus(game, corpus, train_level)
    transfer = None
    for lv in sorted(l for l in counts if l > train_level):
        cand = LevelEvidence.from_corpus(game, corpus, lv)
        if cand is not None and len(cand.actions) >= MIN_TRANSFER_KEYS:
            transfer = lv
            break
    v1 = os.path.join(V1, f"{game}.json")
    if os.path.exists(v1):                                     # held equal to Tier 0a
        with open(v1) as f:
            old = json.load(f)
        was = (old["level1"]["level"] - 1, old["level2"]["level"] - 1 if old.get("level2") else None)
        if was != (train_level, transfer):
            raise RuntimeError(f"{game}: levels {train_level, transfer} differ from Tier 0a's {was}")
    return train, transfer, counts


# --------------------------------------------------------------------------------
# One candidate rule on one level
# --------------------------------------------------------------------------------
def evaluate(code, group, ev, ev_path, prefer=None):
    """Fit a rule's PARAMS to the level and grade it there. Returns the arm."""
    arm = {"code": code, "group": group, "lint": lint(code)}
    found = B.search(code, ev_path, family=B.family_of(group), prefer=prefer)
    arm.update(stage=found["stage"], reason=found.get("reason", "ok"))
    if found["stage"] != "checked":
        return arm
    best, flag = B.choose(found)
    final = graded_final(found, best, ev)
    if final is None:                                          # the level has no binding to try
        arm.update(stage="unbound")
        return arm
    _, params = C.read_params(code)
    res = final["res"]
    arm.update(rule=found["rule"], version=found["version"], has_params=bool(params), binding=final["binding"],
               exact_binding=best is not None, flag=flag,
               binding_text=R2.binding_text(final["binding"], params, ev.api.layout),
               search={k: found[k] for k in ("bindings", "untried", "halving", "exact", "exact_applying",
                                             "guess", "guess_note", "unused")},
               res=res, grade=grade(res, ev, group), trust=K.trust([(res, ev)]))
    arm["reason"] = arm["trust"]["reason"]
    return arm


def graded_final(found, best, ev):
    """The binding a rule is graded and shown with: the chosen exact one, or
    else the inexact one that does best on the level (an exact binding that
    applies to nothing says nothing about the rule)."""
    if best is not None:
        return best
    finals = found.get("finals") or []
    applying = [f for f in finals if f["res"]["applies"].any()]
    if applying:
        return max(applying, key=lambda f: (K.trust([(f["res"], ev)])["claimed_exactness"],
                                            int(f["res"]["cells_right"].sum()), -f["order"]))
    return finals[0] if finals else None


def in_pool(arm):
    """Worth refining: it runs, covers 20 moves and gets more changes right than wrong."""
    return (arm["stage"] == "checked" and arm["trust"]["moves"] >= ADMIT["min_moves"]
            and arm["trust"]["gain_cells"] > 0)


def is_trusted(arm):
    return arm["stage"] == "checked" and arm["trust"]["trusted"]


def carries(arm):
    """Trusted and free of level facts: the rule a group's search stops at."""
    return is_trusted(arm) and not (STOP_NEEDS_CLEAN_LINT and arm["lint"])


def ask(client, parts, instruction, n, seed, group, ev, ev_path, seen):
    """One request for n answers, each with one code-only retry if it ran out
    of room and one repair attempt if its code cannot run. Returns the arms."""
    conv = R2.conversation(parts, instruction)
    outs = client.complete(conv, n=n, max_tokens=R.MAX_TOKENS[True], thinking=True, seed=seed, **R.SAMPLING[True])
    with client.lock:                                          # how long the request really took (the cache keeps it)
        took = next((r.get("seconds", 0.0) for r in client.cache.values() if r["outputs"] is outs), 0.0) / max(n, 1)

    def one(item):
        k, o = item
        tokens, finished, retried = o["tokens"], o["finish"] == "stop", False
        code = R.extract_code(o["text"])
        arm = (evaluate(code, group, ev, ev_path) if code else
               {"code": None, "group": group, "stage": "no-code",
                "reason": "no ```python block in the answer" + ("" if finished else " (ran out of tokens)")})
        if (code is None and not finished) or arm["stage"] in BROKEN:
            msg = R2.FINISH if code is None else R2.REPAIR.format(reason=arm["reason"])
            o2 = client.complete(conv + [{"role": "assistant", "content": R.tail(o["text"])},
                                         {"role": "user", "content": msg}],
                                 n=1, max_tokens=R.MAX_TOKENS[False], thinking=False, seed=seed + 50,
                                 **R.SAMPLING[False])[0]
            tokens, retried = tokens + o2["tokens"], True
            code2 = R.extract_code(o2["text"])
            if code2:
                arm = evaluate(code2, group, ev, ev_path)
        arm.update(tokens=tokens, finished=finished, retried=retried, k=k, llm_seconds=round(took, 1))
        return arm
    with ThreadPoolExecutor(max(1, min(n, 4))) as ex:
        arms = list(ex.map(one, enumerate(outs)))
    for arm in arms:                                           # the same code twice is one rule
        arm["duplicate"] = bool(arm["code"] and arm["code"] in seen)
        if arm["code"]:
            seen.add(arm["code"])
    return arms


def pick(pool, rng):
    """Thompson sampling, as in CP0: the arm with the highest Beta draw."""
    draws = [rng.beta(1 + BANDIT_C * a["trust"]["claimed_exactness"],
                      1 + BANDIT_C * (1 - a["trust"]["claimed_exactness"]) + a["refined"]) for a in pool]
    return int(np.argmax(draws))


def as_member(arm, layout_path, binding=None, **kw):
    return dict(src=arm["code"], binding=arm["binding"] if binding is None else binding,
                layout_path=layout_path, group=arm["group"], **kw)


def by_merit(arms):
    """Best first for the book: rules without level facts, then by gain."""
    return sorted(arms, key=lambda a: (bool(a["lint"]), -a["trust"]["gain_cells"], a["order"]))


def run_group(game, group, ev, ev_path, client, opts):
    """Rounds for one action group (module doc, step 1). Returns its record."""
    t0 = time.time()
    seen, arms, steps = set(), [], []

    def add(new, **kw):
        for a in new:
            a.update(order=len(arms), refined=0, **kw)
            arms.append(a)

    parts, instruction = R2.first_round(ev, group, rng_for(game, group, "prompt"))
    add(ask(client, parts, instruction, opts.k, 1, group, ev, ev_path, seen), origin="first round", step=0)
    rng = rng_for(game, group, "bandit")
    stopped = None
    while stopped is None:
        live = [a for a in arms if not a["duplicate"]]
        book = by_merit([a for a in live if carries(a)])
        left_calls = opts.calls - len(steps)
        if book:                                               # the group has a rule that carries: residual prompts
            if not opts.residual:
                stopped = "first trusted rule"
                break
            comp = K.compose([as_member(a, ev_path) for a in book], ev_path, select=True)
            left = K.unexplained(comp, ev, group) if comp["stage"] == "checked" else {}
            if len(left) < K.MIN_NEW_CASES:
                stopped = f"trusted, with unexplained cells left on {len(left)} cases (fewer than {K.MIN_NEW_CASES})"
                break
            if not left_calls:
                stopped = f"out of calls; trusted, with unexplained cells left on {len(left)} cases"
                break
            parts, instruction, shown = R2.residual(ev, group, [book[i] for i in comp["book"]], left,
                                                    rng_for(game, group, "residual", len(steps) + 1))
            kind, parent = "residual", None
        elif not left_calls:
            stopped = "out of calls; no trusted rule" + (" without level facts" if any(is_trusted(a) for a in live) else "")
            break
        else:
            pool = [a for a in live if in_pool(a)] or [a for a in live if a["stage"] == "checked" and a["trust"]["moves"]]
            if pool:
                parent = pool[pick(pool, rng)]
                parts, instruction, shown = R2.refinement(ev, group, parent, rng_for(game, group, "prompt"),
                                                          rng_for(game, group, "cases", len(steps) + 1))
                parent["refined"] += 1
                kind = "refine"
            else:                                              # nothing has run yet: ask again, with the reasons
                parts, instruction = R2.retry(ev, group, [a["reason"] for a in arms], rng_for(game, group, "prompt"))
                kind, parent, shown = "retry", None, 0
        step = len(steps) + 1
        child = ask(client, parts, instruction, 1, 100 + step, group, ev, ev_path, seen)[0]
        add([child], origin=kind, step=step, parent=None if parent is None else parent["order"], shown=shown)
        steps.append(child)
        print(f"[{game} {group}] call {step} ({kind}"
              + (f" of rule {parent['order']}" if parent is not None else "") + f"): {describe_arm(child)}", flush=True)
    keys = ev.keys_of(group)
    return {"group": group, "moves": int(ev.count[keys].sum()), "cases": int(len(keys)),
            "changing_cases": int((ev.changed[keys] & ~ev.frozen[keys]).sum()),
            "conflict_cases": int((ev.n_variants[keys] > 1).sum()), "calls": len(steps), "stopped": stopped,
            "trusted": sum(is_trusted(a) and not a["duplicate"] for a in arms),
            "carrying": sum(carries(a) and not a["duplicate"] for a in arms),
            "seconds": round(time.time() - t0), "arms": arms}
# In an arm, `order` numbers the rules of its GROUP (and `parent` is such a number);
# `id`, given once the game is done, numbers them across the game.


def describe_arm(a):
    if a["stage"] != "checked":
        return f"{a['stage']} ({a['reason'][:70]})"
    t = a["trust"]
    return (f"{t['claimed_exactness']:.1%} claimed-exact on {t['moves']} moves, {t['cells_right']} cells right"
            + (", TRUSTED" if t["trusted"] else "") + (f", {len(a['lint'])} lint warnings" if a["lint"] else "")
            + f"; {a.get('tokens', 0)} tokens")


# --------------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------------
def rebind(arm, train, fit, p_fit):
    """Fit one book rule's PARAMS to the transfer level's fit split and decide
    whether it is in T1. Returns {in_t1, flag, why, binding, text, wrong, trust}."""
    out = {"in_t1": False, "flag": None, "why": None, "binding": None, "wrong": []}
    found = B.search(arm["code"], p_fit, family=B.family_of(arm["group"]), prefer=arm["binding"])
    if found["stage"] != "checked":
        out.update(flag="binding failed", why=f"it could not be run on the new level: {found['reason']}",
                   text=f"it could not be run: {found['reason']}")
        return out
    _, params = C.read_params(arm["code"])
    best, flag = B.choose(found)
    shown = graded_final(found, best, fit)
    btext = R2.binding_text(shown["binding"], params, fit.api.layout) if shown else "none"
    out.update(flag=flag, search={k: found[k] for k in ("bindings", "untried", "exact", "exact_applying", "guess")})
    if best is None:
        applied = bool(shown and shown["res"]["applies"].any())
        out.update(why="no binding is exact on the fit split" if applied
                   else "it applies to nothing in the fit split and there is no bind() guess to go on",
                   wrong=(shown["res"]["wrong"] if shown else []),
                   text=(f"no values of PARAMS make it right on every one of the first {FIT_MOVES} moves. With the "
                         f"best found ({btext}) it applies to {int(shown['res']['applies'].sum())} cases and its claimed "
                         f"cells are wrong on {int((shown['res']['applies'] & ~shown['res']['claimed_ok']).sum())} of them."
                         if applied else
                         f"it applies to none of the first {FIT_MOVES} moves with any values of PARAMS, and it has no "
                         "bind() to guess them from the first board, so they could not be fitted."
                         if shown else f"PARAMS could not be fitted: {found['reason']}"))
    else:
        t = K.trust([(arm["res"], train), (best["res"], fit)])
        n_app = int(best["res"]["applies"].sum())
        out.update(binding=best["binding"], trust={k: v for k, v in t.items() if k != "levels"},
                   text=(f"with PARAMS fitted to its first {FIT_MOVES} moves ({btext}) it applies to {n_app} cases and "
                         "its claimed cells are right on all of them." if n_app else
                         f"it applies to none of the first {FIT_MOVES} moves, so nothing about it could be checked "
                         f"({btext})."))
        if not t["trusted"]:
            out.update(why=f"not trusted across both levels: {t['reason']}")
        else:
            out["in_t1"] = True
    if arm["lint"]:                                            # level-bound: out of T1 whatever the search found
        out.update(in_t1=False, flag="level-bound rule", why="the lint finds level facts in its code",
                   text=out["text"] + " It was not carried over, because it has level facts written into its code.")
    return out


def score_book(members, p_score, score, base):
    """A book on the score split: its numbers, per-move rates and interval."""
    comp = K.compose(members, p_score)
    if comp["stage"] != "checked":
        raise RuntimeError(f"the book could not be composed: {comp.get('reason')}")
    nums = K.numbers(comp, score)
    gate = gate_interval(score, dict(base, book=moves_right(score, comp["hash"])))
    return dict(nums, per_move=gate, rules_in=len(members),
                clashing_rules=[i for i, c in enumerate(comp["clashes"]) if c])


def slim(arm):
    keep = ("id", "code", "rule", "group", "stage", "reason", "order", "origin", "step", "parent", "k", "tokens",
            "finished", "retried", "duplicate", "refined", "shown", "binding", "exact_binding", "flag", "search",
            "version", "has_params", "lint", "llm_seconds", "t1", "in_book")
    out = {k: v for k, v in arm.items() if k in keep}
    if "trust" in arm:
        out["trust"] = arm["trust"]
        out["grade"] = dict(arm["grade"], missed=len(arm["grade"].get("missed") or []))
    return out


def prepare_game(game, corpus, out):
    """A game's training evidence and the action groups that get calls."""
    t0 = time.time()
    train, transfer_level, counts = pick_levels(game, corpus)
    p_train = os.path.join(out, f"{game}_train.npz")
    train.save(p_train)
    groups = train.groups[:MAX_GROUPS]
    changing = [g for g in groups if (train.changed[train.keys_of(g)] & ~train.frozen[train.keys_of(g)]).any()]
    res = {"game": game, "corpus": corpus, "moves_per_level": {str(k + 1): v for k, v in sorted(counts.items())},
           "train": train.summary(), "groups_dropped": len(train.groups) - len(groups),
           "never_changing": {g: train.group_moves[g] for g in groups if g not in changing}}
    print(f"[{game}] training level {train.level + 1}: {res['train']['moves']} moves, {res['train']['keys']} cases; "
          f"{len(changing)} groups get calls, {len(groups) - len(changing)} never change the board", flush=True)
    return {"game": game, "corpus": corpus, "out": out, "train": train, "p_train": p_train,
            "transfer_level": transfer_level, "changing": changing, "res": res, "t0": t0}


def run_game(game, corpus, out, client, opts):
    """One game from start to end (the tests, and --games with one game)."""
    prep = prepare_game(game, corpus, out)
    records = []
    if client is not None:
        with ThreadPoolExecutor(GROUP_WORKERS) as ex:
            records = list(ex.map(lambda g: run_group(game, g, prep["train"], prep["p_train"], client, opts),
                                  prep["changing"]))
    return finish_game(prep, records, client, opts)


def finish_game(prep, records, client, opts):
    """The book, transfer and the per-game record, once every group is done.
    `records` are run_group's, in the order of prep['changing']."""
    game, corpus, out, train, p_train = (prep[k] for k in ("game", "corpus", "out", "train", "p_train"))
    transfer_level, changing, res, t0 = prep["transfer_level"], prep["changing"], prep["res"], prep["t0"]
    p_fit, p_score = (os.path.join(out, f"{game}_{k}.npz") for k in ("fit", "score"))
    arms = [a for r in records for a in r["arms"]]
    # -- the book, on the training level ------------------------------------------------------------
    order = {g: i for i, g in enumerate(changing)}
    trusted = sorted((a for a in arms if is_trusted(a) and not a["duplicate"]),
                     key=lambda a: (order[a["group"]], bool(a["lint"]), -a["trust"]["gain_cells"], a["order"]))
    comp = K.compose([as_member(a, p_train) for a in trusted], p_train, select=True)
    if comp["stage"] != "checked":
        raise RuntimeError(f"{game}: the book could not be composed: {comp.get('reason')}")
    book = [trusted[i] for i in comp["book"]]
    for i, a in enumerate(trusted):
        a["in_book"] = i in comp["book"] or comp["notes"][i]
    res["book_train"] = K.numbers(comp, train)
    print(f"[{game}] book: {len(book)} of {len(trusted)} trusted rules; training level: {res['book_train']}", flush=True)
    # -- transfer ------------------------------------------------------------------------------------
    res["transfer"] = None
    if transfer_level is not None:
        fit, score = LevelEvidence.split(game, corpus, transfer_level, FIT_MOVES, known_mask=train.mask)
        fit.save(p_fit)
        res.update(transfer_level=transfer_level + 1, fit=fit.summary())
        if score is not None:
            score.save(p_score)
            res["score"] = score.summary()
            base = {"nothing": moves_right(score, nothing_hash(score)),
                    "memory": moves_right(score, memory_hash([train, fit], score))}
            tr = {"t0": score_book([as_member(a, p_train) for a in book], p_score, score, base)}
            with ThreadPoolExecutor(GROUP_WORKERS) as ex:
                for a, t1 in zip(book, ex.map(lambda a: rebind(a, train, fit, p_fit), book)):
                    a["t1"] = t1
            kept = [a for a in book if a["t1"]["in_t1"]]
            tr["t1"] = score_book([as_member(a, p_fit, a["t1"]["binding"]) for a in kept], p_score, score, base)
            tr["t1"]["flags"] = dict(collections.Counter(a["t1"]["flag"] or "rebound" for a in book))
            untested = [a for a in kept if a["t1"]["flag"] == "binding untested"]
            tr["t1_without_untested"] = score_book(
                [as_member(a, p_fit, a["t1"]["binding"]) for a in kept if a not in untested], p_score, score, base)
            # T2: one two-level repair call for each group that lost a rule
            lost = {}
            for a in book:
                if not a["t1"]["in_t1"]:
                    lost.setdefault(a["group"], a)             # the group's best rule that did not carry
            repaired = []
            if client is not None and lost:
                def repair(a):
                    t1 = a["t1"]
                    parts, instruction = R2.two_level(train, fit, a["group"], a, t1, rng_for(game, a["group"], "two-level"))
                    child = ask(client, parts, instruction, 1, 200, a["group"], train, p_train, set())[0]
                    child.update(refined=0, origin="two-level", step=0, parent=a["order"], duplicate=False)
                    if is_trusted(child):
                        child["t1"] = rebind(child, train, fit, p_fit)
                    return child
                with ThreadPoolExecutor(GROUP_WORKERS) as ex:
                    repaired = list(ex.map(repair, lost.values()))
                for c in repaired:
                    c["order"] = sum(1 for a in arms if a["group"] == c["group"])
                    arms.append(c)
            good = [c for c in repaired if c.get("t1", {}).get("in_t1")]
            sel = K.compose([as_member(a, p_train, locked=True) for a in kept] + [as_member(c, p_train) for c in good],
                            p_train, select=True)
            t2 = (kept + good) if sel["stage"] != "checked" else [(kept + good)[i] for i in sel["book"]]
            tr["t2"] = score_book([as_member(a, p_fit, a["t1"]["binding"]) for a in t2], p_score, score, base)
            tr["t2"].update(repair_calls=len(repaired), repaired=len(good))
            for a in book + repaired:
                if "t1" in a:
                    a["t1"] = {k: v for k, v in a["t1"].items() if k != "wrong"}
            res["transfer"] = tr
            g = tr["t1"]["per_move"]
            print(f"[{game}] transfer level {transfer_level + 1}: T1 {g['rate']} with {tr['t1']['rules_in']} of "
                  f"{len(book)} rules; beats both: {g['beats_both']}; wrong on {tr['t1']['wrong']:.1%} of changing cases",
                  flush=True)
    for i, a in enumerate(arms):
        a["id"] = i
    res["groups"] = [dict(r, arms=[a["id"] for a in r["arms"]]) for r in records]
    res["candidates"] = [slim(a) for a in arms]
    res["book"] = [a["id"] for a in book]
    answers = [a for a in arms if "tokens" in a]
    res["llm"] = {"answers": len(answers), "tokens": sum(a["tokens"] for a in answers),
                  "ran_out": sum(not a["finished"] for a in answers), "retried": sum(a["retried"] for a in answers),
                  "untried_bindings": sum((a.get("search") or {}).get("untried", 0) for a in arms)}
    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else list(o))
    return res


# --------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------
def _iv(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d["lo"] is not None else f"{d['point']:+.3f} (no interval)"


def verdict(results):
    """(primary games, unsafe games, PASS or FAIL) from the per-game results."""
    scored = [r for r in results if r.get("transfer")]
    primary = [r["game"] for r in scored if r["transfer"]["t1"]["per_move"]["beats_both"]]
    unsafe = [r["game"] for r in scored if r["transfer"]["t1"]["wrong"] > GATE_WRONG]
    unsafe += [r["game"] for r in results if not r.get("transfer")]          # "on every game": an unscored game fails it
    return primary, unsafe, "PASS" if len(primary) >= GATE_GAMES and not unsafe else "FAIL"


def report(out, results, opts, client):
    primary, unsafe, word = verdict(results)
    lines = ["# Rulebook v2, CP1: Tier 0a v2 - rules that carry to the next level", "",
          f"Pre-registered in docs/plans/rulebook-v2-prereg.md section 4. Games: {', '.join(r['game'] for r in results)}; "
          f"one recorded mb_gated_att run each (seed {opts.seed}, `{opts.manifest}`). {opts.k} first-round candidates "
          f"and up to {opts.calls} more calls per action group. Tuned on dev games.", "",
          "Built: PARAMS and re-binding by search, the literal lint, claimed cells and the composed book, trust on "
          "every level seen, T0/T1/T2, " + ("residual prompts" if opts.residual else "NO residual prompts") +
          ". NOT built (the plan's cut lines): the simplification pass, api.t."]
    if client is not None:
        m = client.meter
        answers = sum(r["llm"]["answers"] for r in results)
        ran_out = sum(r["llm"]["ran_out"] for r in results)
        lines += [f"Model `{client.served_model()}`, thinking on. LLM: {m['requests']} requests ({m['cached']} from cache), "
               f"{m['completions']} new answers, {m['tokens']:,} tokens, {m['seconds'] / 60:.0f} request-minutes. "
               f"{ran_out} of {answers} answers ran out of room ({ran_out / max(answers, 1):.0%})."]
        lines += ["", "## The gate", "",
               f"**Primary:** T1 beats both baselines with both intervals above zero on **{len(primary)} of {len(results)} "
               f"games** ({', '.join(primary) or 'none'}); needs {GATE_GAMES}.",
               f"**Safety:** T1 wrong on more than {GATE_WRONG:.0%} of the transfer level's changing cases on "
               f"**{len(unsafe)} games** ({', '.join(unsafe) or 'none'}); needs none.", "",
               f"**{word}** (pass = primary and safety)."]
    untried = sum(r["llm"]["untried_bindings"] for r in results)
    if untried:
        lines += ["", f"NOTE: the binding search ran out of time on {untried} bindings in all; those were never called exact."]
    for name, title in (("t1", "T1: the book after re-binding, no LLM (the gate's arm)"),
                        ("t0", "T0: the book untouched"), ("t2", "T2: after one repair call per group that lost a rule"),
                        ("t1_without_untested", "T1 without the rules the fit split could not test (reported only)")):
        lines += ["", f"## {title}", "",
               "Per recorded move of the score split; differences are book minus baseline with the 95% interval "
               "(pre-registration 4.4). 'Wrong' is the share of distinct changing cases with a wrong claimed cell.", "",
               "| game | level | moves | resampled | rules | book | nothing | memory | book - nothing | book - memory | "
               "beats both | wrong | whole board right (cases) | cell coverage |",
               "|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|"]
        for r in results:
            t = (r.get("transfer") or {}).get(name)
            if not t:
                lines.append(f"| {r['game']} | - | - | - | - | - | - | - | - | - | - | - | - | - |")
                continue
            g = t["per_move"]
            lines.append(f"| {r['game']} | {r['transfer_level']} | {g['moves']:,} | {g['units']} {g['unit']} | {t['rules_in']} | "
                      f"{g['rate']['book']:.3f} | {g['rate']['nothing']:.3f} | {g['rate']['memory']:.3f} | "
                      f"{_iv(g['diff']['nothing'])} | {_iv(g['diff']['memory'])} | {'yes' if g['beats_both'] else 'no'} | "
                      f"{t['wrong']:.1%} | {t['case_coverage']:.1%} | {t['cell_coverage']:.1%} |")
    lines += ["", "## v1, for comparison (tools/wm_baseline.py, the same moves and interval)", ""]
    rows = []
    for r in results:
        p = os.path.join(V1_ROW, f"{r['game']}.json")
        if os.path.exists(p):
            with open(p) as f:
                v = json.load(f)
            if "t0_moves" in v:
                m = v["t0_moves"]
                rows.append(f"| {r['game']} | {m['rate']['book']:.3f} | {_iv(m['diff']['nothing'])} | "
                            f"{_iv(m['diff']['memory'])} | {'yes' if m['beats_both'] else 'no'} | "
                            f"{v['t0']['wrong_on_changing']:.1%} |")
    lines += (["| game | v1's book | book - nothing | book - memory | beats both | wrong |", "|---|---:|---|---|---|---:|"] + rows
           if rows else ["(run tools/wm_baseline.py first)"])
    covered = [r["game"] for r in results if r["book_train"]["cell_coverage"] >= COVERAGE_TARGET]
    lines += ["", "## Training level: the trusted book (reported, not gating)", "",
           f"Claimed-cell trusted coverage >= {COVERAGE_TARGET:.0%} on {len(covered)} of {len(results)} games "
           f"({', '.join(covered) or 'none'}); the target was 3.", "",
           "| game | level | moves | cases | groups with calls | never-changing groups | book rules | "
           "claimed-cell exactness | cell coverage | strict case coverage | answers | ran out | tokens |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        b, e, m = r["book_train"], r["train"], r["llm"]
        lines.append(f"| {r['game']} | {e['level']} | {e['moves']:,} | {e['keys_kept']:,} | {len(r['groups'])} | "
                  f"{len(r['never_changing'])} | {b['rules']} | {b['claimed_exactness']:.1%} | {b['cell_coverage']:.1%} | "
                  f"{b['case_coverage']:.1%} | {m['answers']} | {m['ran_out']} | {m['tokens']:,} |")
    for r in results:
        lines += ["", f"## {r['game']}", "",
               "| action group | moves | changing cases | calls after round 1 | trusted rules | of them without level facts | "
               "why it stopped |", "|---|---:|---:|---:|---:|---:|---|"]
        for g in r["groups"]:
            lines.append(f"| {g['group']} | {g['moves']:,} | {g['changing_cases']} | {g['calls']} | {g['trusted']} | "
                      f"{g['carrying']} | {g['stopped']} |")
        if r["never_changing"]:
            lines += ["", "No calls (the board never changed): " +
                   ", ".join(f"{g} ({n:,} moves)" for g, n in r["never_changing"].items()) + "."]
        lines += ["", "The book, and what happened to each rule on the next level:", ""]
        for i in r["book"]:
            c = r["candidates"][i]
            t1 = c.get("t1") or {}
            fate = ("carried" + (f" ({t1['flag']})" if t1.get("flag") else "") if t1.get("in_t1") else
                    f"left out: {t1.get('flag') or 'not trusted'} - {t1.get('why')}" if t1 else "no transfer level")
            lines.append(f"- **{c['rule']}** ({c['group']}; {c['trust']['moves']} moves, {c['trust']['cells_right']} cells "
                      f"right; binding {c['binding'] or 'none'}): {fate}")
        if not r["book"]:
            lines.append("(no trusted rule)")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nwrote {out}/report.md")


def run_all(games, runs, out, client, opts):
    """Every game, with ALL their action groups in one pool of GROUP_WORKERS, so
    the server stays busy across games (one game at a time left it half idle on
    games with one or two groups). A game's book and transfer start as soon as
    its last group is done. Only the order of the calls changes: each group's
    calls and seeds are its own. Returns (results in game order, failed games)."""
    import traceback
    from concurrent.futures import as_completed
    preps, failed = {}, []
    for g in games:
        try:
            preps[g] = prepare_game(g, runs[g], out)
        except Exception:
            traceback.print_exc()
            failed.append(g)
            print(f"[{g}] FAILED - see the traceback above", flush=True)
    done = {g: [None] * len(p["changing"]) for g, p in preps.items()}
    waiting = {g: len(p["changing"]) for g, p in preps.items()}
    bad, finished = set(), {}
    with ThreadPoolExecutor(GROUP_WORKERS) as groups_pool, ThreadPoolExecutor(2) as finish_pool:
        def finish(g):
            if g in bad:
                return
            records = [r for r in done[g] if r is not None]       # none at all without an LLM
            finished[g] = finish_pool.submit(finish_game, preps[g], records, client, opts)
        if client is None:
            for g in preps:
                finish(g)
        else:
            jobs = {groups_pool.submit(run_group, g, grp, preps[g]["train"], preps[g]["p_train"], client, opts): (g, i)
                    for g in preps for i, grp in enumerate(preps[g]["changing"])}
            for g in preps:
                if not waiting[g]:
                    finish(g)
            for f in as_completed(jobs):
                g, i = jobs[f]
                try:
                    done[g][i] = f.result()
                except Exception:
                    traceback.print_exc()
                    bad.add(g)
                    print(f"[{g}] a group FAILED - see the traceback above", flush=True)
                waiting[g] -= 1
                if not waiting[g]:
                    finish(g)
        results = []
        for g in preps:
            if g in bad or g not in finished:
                failed.append(g)
                continue
            try:
                results.append(finished[g].result())
            except Exception:                 # keep the other games; this one resumes from the cache
                traceback.print_exc()
                failed.append(g)
                print(f"[{g}] FAILED - see the traceback above", flush=True)
    return results, sorted(set(failed), key=games.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--games", default=",".join(tiers.DEV))
    ap.add_argument("--seed", type=int, default=0, help="which recorded run of each game")
    ap.add_argument("--k", type=int, default=K_FIRST)
    ap.add_argument("--calls", type=int, default=CALLS)
    ap.add_argument("--no-residual", dest="residual", action="store_false",
                    help="stop a group at its first trusted rule (no residual prompts)")
    ap.add_argument("--evidence-only", action="store_true", help="no LLM: evidence, baselines and intervals, empty book")
    ap.add_argument("--from-cache", action="store_true",
                    help="re-score a finished run from its answer cache: no server is contacted, and a request "
                         "that is not in the cache is an error, never a new answer")
    ap.add_argument("--url", default="http://127.0.0.1:8018/v1")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rulebook", "v2", "tier0a_v2"))
    a = ap.parse_args()
    inside = os.path.relpath(os.path.abspath(a.out), ROOT) + os.sep
    frozen = [e["name"] for e in registry.load()["artifacts"] if any(f.startswith(inside) for f in e["files"])]
    if frozen:
        sys.exit(f"{a.out} holds the registered result {frozen[0]!r}; write somewhere else")
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for row in read_manifest(os.path.join(ROOT, a.manifest), strict=True):
        rundir = row["run_dir"]
        if int(row["seed"]) == a.seed:
            runs[row["game"]] = os.path.join(rundir if os.path.isabs(rundir) else os.path.join(ROOT, rundir),
                                             "transitions")
    client = None if a.evidence_only else ChatClient(a.url, cache_path=os.path.join(a.out, "llm_cache.jsonl"))
    if client is not None and a.from_cache:
        models = {r["model"] for r in client.cache.values()}
        if len(models) != 1:
            sys.exit(f"--from-cache needs a cache from exactly one model, found {sorted(models)}")
        client._served = models.pop()

        def refuse(messages, n):
            raise RuntimeError("an answer is not in the cache: the run did not follow the cached path")
        client.mock = refuse
    if client is not None:
        print(f"model: {client.served_model()}" + (" (answers from the cache only)" if a.from_cache else ""), flush=True)
    results, failed = run_all(a.games.split(","), runs, a.out, client, a)
    if failed:
        # No report and no gate verdict on an incomplete set of games.
        sys.exit(f"{len(failed)} game(s) failed: {', '.join(failed)}. Finished games are saved in "
                 f"{a.out}; rerun the same command to resume from the answer cache.")
    report(a.out, results, a, client)


if __name__ == "__main__":
    main()
