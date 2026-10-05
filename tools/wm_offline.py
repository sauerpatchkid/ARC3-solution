#!/usr/bin/env python3
"""wm_offline.py - Rulebook Tier 0a: the offline rule test (docs/plans/llm-rulebook.md
sections 6.1 and 6.4, gate G1). No agent changes: recorded mb_gated_att runs only.

For each dev game, from ONE recorded run (seed 0 of the B0 confirm):
  0. LEVELS. The TRAINING level is the run's first level with at least 500
     recorded moves (level 1 for seven of the dev games; cd82 finished level 1
     in 9 moves, so it trains on level 2 - the plan writes rules only once
     enough evidence has accumulated). The TRANSFER level is the next level
     after it with at least 50 distinct keys.
  1. RULES. The training level's moves become the evidence (wm.evidence). For each action
     group the LLM writes k candidate rules from up to 6 recorded moves; every
     candidate is checked on every key of the level; round 2 (with feedback) runs
     only for groups that still have nothing usable.
  2. COVERAGE. The rule book = per group, the best plan-eligible rule (right on
     every key it applies to) and the best known no-op. Coverage = the share of
     the training level's distinct CHANGING keys the book predicts exactly;
     UNKNOWN and wrong are both misses.
  3. TRANSFER. The same book, unrepaired, on the transfer level's keys, beside
     the "nothing changes" and "memory" baselines.

  experiments/rulebook/serve.sh &                       # the LLM, multimodal, port 8018
  uv run python tools/wm_offline.py                     # 8 dev games
  uv run python tools/wm_offline.py --evidence-only     # no LLM: evidence + baselines
  uv run python tools/wm_offline.py --games ft09 --out results/rulebook/tier0a_smoke

PRE-REGISTERED (2026-10-05, before any Tier 0a LLM answer; the plan's G1):
  Games tu93 tr87 dc22 g50t vc33 ft09 m0r0 cd82; run = seed 0 of
  results/confirm_upgrade/20260927_221915; model cyankiwi/Qwen3.8-27B-AWQ-INT4,
  thinking on, k = 4 candidates, at most 2 rounds, 20,480-token answers with a
  code-only retry and one repair attempt, at most 12 action groups per game (the
  most recorded), check set capped at 40,000 keys per level.
  G1 = PASS when training-level coverage (plan-eligible book, UNKNOWN = miss) is
  >= 80% on at least 3 of the 8 games. Reported beside it, not part of the gate:
  the coverage of the 95%-admitted book, conflicts, and transfer.
  The level rule in step 0 was fixed on 2026-10-05 after seeing only how many
  moves each level has (no LLM answer); the plan's G1 says "level 1".
  If G1 fails: one day to try thinking off / other settings on these dev games,
  then narrow the build to the games and groups that pass.
Writes <out>/{<game>.json, report.md, llm_cache.jsonl} and the evidence files.
"""
import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
from wm import rules as R  # noqa: E402
from wm.check import (book_status, check_rule, coverage, grade, pick_book,  # noqa: E402
                      transfer)
from wm.evidence import LevelEvidence  # noqa: E402
from wm.llm import ChatClient  # noqa: E402

MANIFEST = "results/confirm_upgrade/20260927_221915/manifest.tsv"
DEV_GAMES = ["tu93", "tr87", "dc22", "g50t", "vc33", "ft09", "m0r0", "cd82"]
K, ROUNDS, MAX_GROUPS = 4, 2, 12
G1_COVERAGE, G1_GAMES = 0.80, 3
MIN_TRAIN_MOVES, MIN_TRANSFER_KEYS = 500, 50
CHECK_WORKERS, LLM_WORKERS = 8, 4


def rng_for(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def level_moves(corpus):
    """{level: recorded moves} for one run."""
    import collections
    import glob
    c = collections.Counter()
    for p in sorted(glob.glob(os.path.join(corpus, "shard_*.npz"))):
        with np.load(p) as z:
            for lv, n in zip(*np.unique(z["levels"], return_counts=True)):
                c[int(lv)] += int(n)
    return c


def needs_more(ev, group, cands):
    """Round 2 only where round 1 left nothing usable: no admitted rule for a
    group whose moves change the board, or no known no-op for one whose moves
    never do."""
    keys = ev.keys_of(group)
    changes = bool((ev.changed[keys] & ~ev.frozen[keys]).any())
    mine = [c for c in cands if c["group"] == group and c["stage"] == "checked"]
    if changes:
        return not any(c["grade"]["admitted"] for c in mine)
    return not any(c["grade"]["known_noop"] for c in mine)


def run_game(game, corpus, out, client, a):
    t0 = time.time()
    counts = level_moves(corpus)
    train_level = next((lv for lv in sorted(counts) if counts[lv] >= MIN_TRAIN_MOVES), min(counts))
    ev1 = LevelEvidence.from_corpus(game, corpus, train_level)
    ev2 = None
    for lv in sorted(l for l in counts if l > train_level):
        cand = LevelEvidence.from_corpus(game, corpus, lv)
        if cand is not None and len(cand.actions) >= MIN_TRANSFER_KEYS:
            ev2 = cand
            break
    p1, p2 = os.path.join(out, f"{game}_train.npz"), os.path.join(out, f"{game}_transfer.npz")
    ev1.save(p1)
    if ev2 is not None:
        ev2.save(p2)
    groups = ev1.groups[:MAX_GROUPS]
    res = {"game": game, "corpus": corpus, "moves_per_level": {str(k + 1): v for k, v in sorted(counts.items())},
           "level1": ev1.summary(), "level2": ev2.summary() if ev2 is not None else None,
           "groups": {g: {"moves": ev1.group_moves[g],
                          "changing_keys": int(ev1.changed[ev1.keys_of(g)].sum())} for g in groups},
           "groups_dropped": len(ev1.groups) - len(groups)}
    print(f"[{game}] training level {train_level + 1}: {res['level1']['moves']} moves, {res['level1']['keys']} keys, "
          f"{len(groups)} groups, {res['level1']['conflict_keys']} conflict keys, "
          f"mask {res['level1']['mask_cells']} cells", flush=True)

    cands, seen, rounds = [], set(), []
    if client is not None:
        for rnd in range(1, a.rounds + 1):
            todo = groups if rnd == 1 else [g for g in groups if needs_more(ev1, g, cands)]
            if not todo:
                break
            tg = time.time()

            def ask(g):
                parts = R.group_prompt(ev1, g, rng_for(game, g, "prompt"))
                if rnd > 1:
                    parts = parts + R.feedback(cands, ev1, g, rng_for(game, g, rnd))
                conv = R.conversation(parts, thinking=True)
                outs = client.complete(conv, n=a.k, max_tokens=R.MAX_TOKENS[True], thinking=True,
                                       seed=rnd, **R.SAMPLING[True])
                return g, conv, outs
            with ThreadPoolExecutor(LLM_WORKERS) as ex:
                answers = list(ex.map(ask, todo))
            new, retry = [], []
            for g, conv, outs in answers:
                for k, o in enumerate(outs):
                    code = R.extract_code(o["text"])
                    c = {"game": game, "group": g, "round": rnd, "k": k, "order": len(cands),
                         "code": code, "tokens": o["tokens"], "finished": o["finish"] == "stop",
                         "plan": o["text"].split("</think>")[-1].split("```")[0].strip()[:600]}
                    if code is None:
                        c.update(stage="no-code", reason="no ```python block in the answer"
                                 + ("" if c["finished"] else " (ran out of tokens)"))
                        if not c["finished"]:
                            retry.append((c, conv, o["text"]))
                    elif code in seen:
                        c.update(stage="duplicate", reason="same code as an earlier candidate")
                    else:
                        seen.add(code)
                        new.append((c, conv, o["text"]))
                    cands.append(c)
            gen_s, tc = time.time() - tg, time.time()

            def check(c):
                r = check_rule(c["code"], p1)
                c.update(stage=r["stage"], reason=r.get("reason", "ok"))
                if r["stage"] == "checked":
                    c.update(res=r, rule=r["rule"], grade=grade(r, ev1, c["group"]))
                    c["reason"] = c["grade"]["reason"]
                return c
            with ThreadPoolExecutor(CHECK_WORKERS) as ex:
                list(ex.map(check, [c for c, _, _ in new]))

            # one more try, without thinking, for answers that ran out of room or code that failed to run
            broken = [(c, conv, text) for c, conv, text in new
                      if c["stage"] in ("static", "runtime", "timeout")] + retry

            def fix(item):
                c, conv, text = item
                msg = R.FINISH if c["stage"] == "no-code" else R.REPAIR.format(reason=c["reason"])
                rconv = conv + [{"role": "assistant", "content": R.tail(text)},
                                {"role": "user", "content": msg}]
                o = client.complete(rconv, n=1, max_tokens=R.MAX_TOKENS[False], thinking=False,
                                    seed=rnd + 50, **R.SAMPLING[False])[0]
                return c, R.extract_code(o["text"]), o
            fixed = []
            if broken:
                with ThreadPoolExecutor(LLM_WORKERS) as ex:
                    for c, code, o in ex.map(fix, broken):
                        if code is not None and code not in seen:
                            seen.add(code)
                            c2 = {"game": game, "group": c["group"], "round": rnd, "k": c["k"],
                                  "order": len(cands), "code": code, "tokens": o["tokens"],
                                  "finished": True, "plan": c["plan"], "repaired": True}
                            cands.append(c2)
                            fixed.append(c2)
                with ThreadPoolExecutor(CHECK_WORKERS) as ex:
                    list(ex.map(check, fixed))
            this = [c for c in cands if c["round"] == rnd]
            stages = {}
            for c in this:
                stages[c["stage"]] = stages.get(c["stage"], 0) + 1
            rounds.append({"round": rnd, "groups": len(todo), "gen_s": round(gen_s), "check_s": round(time.time() - tc),
                           "candidates": len(this), "stages": stages,
                           "admitted": sum(1 for c in this if c["stage"] == "checked" and c["grade"]["admitted"]),
                           "plan_eligible": sum(1 for c in this if c["stage"] == "checked" and c["grade"]["plan_eligible"]),
                           "known_noop": sum(1 for c in this if c["stage"] == "checked" and c["grade"]["known_noop"])})
            print(f"[{game}] round {rnd}: {len(todo)} groups, {rounds[-1]['candidates']} candidates, {stages}; "
                  f"{rounds[-1]['admitted']} admitted, {rounds[-1]['plan_eligible']} plan-eligible, "
                  f"{rounds[-1]['known_noop']} known no-ops; {gen_s:.0f}s LLM, {time.time() - tc:.0f}s checks",
                  flush=True)

    book, admitted_book = pick_book(cands)
    st1 = book_status([c["res"] for c in book], len(ev1.actions))
    noops = [c for c in book if c["grade"]["known_noop"] and not c["grade"]["admitted"]]
    st1_adm = book_status([c["res"] for c in admitted_book + noops], len(ev1.actions))
    res.update(rounds=rounds, coverage=coverage(st1, ev1), coverage_admitted=coverage(st1_adm, ev1))
    if ev2 is not None:
        r2 = [check_rule(c["code"], p2) for c in book]
        ok2 = [r for r in r2 if r["stage"] == "checked"]
        res["transfer"] = transfer(book_status(ok2, len(ev2.actions)), ev1, ev2)
        for c, r in zip(book, r2):
            c["transfer"] = ({"keys": int(r["applies"].sum()),
                              "right": int((r["applies"] & r["correct"]).sum())}
                             if r["stage"] == "checked" else {"failed": r.get("reason", r["stage"])})
    else:
        res["transfer"] = None
    keep = ("group", "round", "k", "order", "stage", "reason", "rule", "code", "plan", "tokens",
            "finished", "repaired", "grade", "transfer")
    slim = lambda c: {k: (dict(v, missed=len(v.get("missed", []))) if k == "grade" else v)
                      for k, v in c.items() if k in keep}
    res["book"] = [slim(c) for c in book]
    res["admitted_book"] = [slim(c) for c in admitted_book]
    res["candidates"] = [slim(c) for c in cands]
    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(out, f"{game}.json"), "w") as f:
        json.dump(res, f, indent=1)
    cov = res["coverage"]
    print(f"[{game}] book: {len(book)} rules; training-level coverage {cov['coverage']:.0%} of "
          f"{cov['changing_keys']} changing keys (wrong {cov['wrong']:.0%}, unknown {cov['unknown']:.0%})"
          + (f"; transfer level {ev2.level + 1}: book {res['transfer']['book']:.3f} vs nothing "
             f"{res['transfer']['nothing']:.3f}, memory {res['transfer']['memory']:.3f}"
             if res["transfer"] else "; no transfer level"), flush=True)
    return res


def report(out, results, a, client):
    passed = [r["game"] for r in results if r["coverage"]["coverage"] >= G1_COVERAGE]
    L = ["# Rulebook Tier 0a: the offline rule test", "",
         f"Games: {', '.join(r['game'] for r in results)}. One recorded mb_gated_att run each (seed 0, "
         f"`{a.manifest}`). k = {a.k} candidates, at most {a.rounds} rounds."]
    if client is not None:
        m = client.meter
        L += [f"Model `{client.served_model()}`, thinking on. LLM: {m['requests']} requests "
              f"({m['cached']} from cache), {m['completions']} new answers, {m['tokens']:,} tokens, "
              f"{m['seconds'] / 60:.0f} request-minutes.", "",
              "## G1 (pre-registered in tools/wm_offline.py)", "",
              f"Training-level coverage >= {G1_COVERAGE:.0%} on at least {G1_GAMES} of the {len(results)} games: "
              f"**{len(passed)} game(s)** ({', '.join(passed) or 'none'}) -> "
              f"**{'PASS' if len(passed) >= G1_GAMES else 'FAIL'}**"]
    L += ["", "## Training level: evidence and coverage", "",
          "Coverage = share of the level's distinct changing keys the rule book predicts exactly "
          "(trusted rules only; unknown and wrong are misses). 'Admitted' = the looser book of rules "
          "that are at least 95% right.", "",
          "| game | level | moves | keys | changing keys | conflict keys | mask cells | groups | book rules | "
          "**coverage** | wrong | unknown | admitted-book coverage |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        e, c, ca = r["level1"], r["coverage"], r["coverage_admitted"]
        L.append(f"| {r['game']} | {e['level']} | {e['moves']:,} | {e['keys']:,} | {c['changing_keys']:,} | {e['conflict_keys']} | "
                 f"{e['mask_cells']} | {len(r['groups'])} | {len(r['book'])} | **{c['coverage']:.0%}** | "
                 f"{c['wrong']:.0%} | {c['unknown']:.0%} | {ca['coverage']:.0%} |")
    L += ["", "## Transfer level: the same book, before any repair", "",
          "One-step exact rate over the transfer level's distinct keys. 'Book' reads UNKNOWN as "
          "nothing-changes so it is comparable with the baselines; 'coverage' is the strict share of that "
          "level's changing keys.", "",
          "| game | level | keys | book | nothing | memory | beats both | coverage | wrong |",
          "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in results:
        t = r["transfer"]
        L.append(f"| {r['game']} | " + (
            f"{r['level2']['level']} | {t['keys']:,} | {t['book']:.3f} | {t['nothing']:.3f} | {t['memory']:.3f} | "
            f"{'yes' if t['beats_both'] else 'no'} | {t['test_coverage']:.0%} | {t['test_wrong']:.0%} |"
            if t else "- | - | - | - | - | - | - | - |"))
    for r in results:
        L += ["", f"## {r['game']}", ""]
        for rd in r.get("rounds", []):
            L.append(f"- round {rd['round']}: {rd['groups']} groups, {rd['candidates']} candidates {rd['stages']}; "
                     f"{rd['admitted']} admitted, {rd['plan_eligible']} plan-eligible, {rd['known_noop']} known "
                     f"no-ops; {rd['gen_s']} s LLM, {rd['check_s']} s checks")
        L += ["", "| action group | moves | changing keys | checked | admitted | plan-eligible | best rule | right | gain |",
              "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
        for g, info in r["groups"].items():
            cs = [c for c in r["candidates"] if c["group"] == g and c["stage"] == "checked"]
            best = max(cs, key=lambda c: (c["grade"]["plan_eligible"], c["grade"]["gain"], c["grade"]["admitted"],
                                          c["grade"]["accuracy"])) if cs else None
            L.append(f"| {g} | {info['moves']:,} | {info['changing_keys']} | {len(cs)} | "
                     f"{sum(c['grade']['admitted'] for c in cs)} | {sum(c['grade']['plan_eligible'] for c in cs)} | "
                     + (f"{best['rule'][:100]} | {best['grade']['accuracy']:.0%} | {best['grade']['gain']} |"
                        if best else "- | - | - |"))
        L += ["", "Rule book (trusted rules):", ""]
        for c in r["book"] or []:
            g, t = c["grade"], c.get("transfer")
            kind = "no-op" if g["known_noop"] and not g["admitted"] else "rule"
            L.append(f"- [{kind}] **{c['rule']}** ({c['group']}, round {c['round']}): {g['moves']} moves, "
                     f"gain {g['gain']}" + (f"; transfer level: applies to {t['keys']} keys, right on {t['right']}"
                                            if t and "keys" in t else ""))
        if not r["book"]:
            L.append("(none)")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:28]))
    print(f"\nwrote {out}/report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--games", default=",".join(DEV_GAMES))
    ap.add_argument("--seed", type=int, default=0, help="which recorded run of each game")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--evidence-only", action="store_true", help="no LLM: evidence, baselines, empty book")
    ap.add_argument("--url", default="http://127.0.0.1:8018/v1")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rulebook", "tier0a"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for line in open(os.path.join(ROOT, a.manifest)):
        rundir, game, seed, _ = line.rstrip("\n").split("\t")
        if int(seed) == a.seed:
            runs[game] = os.path.join(rundir if os.path.isabs(rundir) else os.path.join(ROOT, rundir), "transitions")
    client = None if a.evidence_only else ChatClient(a.url, cache_path=os.path.join(a.out, "llm_cache.jsonl"))
    if client is not None:
        print(f"model: {client.served_model()}", flush=True)
    results = [run_game(g, runs[g], a.out, client, a) for g in a.games.split(",")]
    report(a.out, results, a, client)


if __name__ == "__main__":
    main()
