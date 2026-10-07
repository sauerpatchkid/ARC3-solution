#!/usr/bin/env python3
"""advice_probe.py - Coach step 4: the offline test (docs/plans/llm-coach.md §7.1).

Does the LLM pick better next experiments than random, and no worse than the
hand rule, on screens from runs we already have? No agent changes: recorded
mb_gated_att runs are replayed through coach.history.History exactly as the live
agent would feed it.

  uv run python legacy/coach_track/tools/advice_probe.py                  # 8 dev games, needs serve.sh running
  uv run python legacy/coach_track/tools/advice_probe.py --no-llm         # build points, score random + rule only

PRE-REGISTERED (written 2026-10-03, before any LLM answer was scored):
  Points, from the B0 confirm runs (mb_gated_att, 100k) of the dev games
  tu93 tr87 dc22 g50t vc33 ft09 m0r0 cd82, seeds 0-2:
    prewin  the screen each level-completing move was made on; target = that move.
    stall   the first moment in a level with >= 1500 moves since the last new
            screen, again after each further 3000 stuck moves, at most 4 per
            level; target = the next move in the same level that reached a new
            screen (unresolved if the level ended first - not scored).
  A target is scored if it is a click on an object listed in the summary
  (object point: hit = that object is in the ranker's top 3) or a button
  (button point: hit = that button has the ranker's top weight; ties broken by
  a fixed per-point random draw). Background clicks and unlisted objects are
  not scored and are reported as coverage.
  Rankers: random (exact expectation: min(3, N)/N for objects, 1/k for k
  buttons), the hand rule (coach.advice.heuristic_advice, arm B2), and the LLM
  (prompt_v1, thinking off, temperature 0.3, seed = point index, 400 tokens).
  An LLM answer that fails to parse counts as a miss.
  G1 = GO when all three hold over the pooled scored points:
    1. at least 15 scored points;
    2. LLM hits > random's expectation, one-sided Poisson-binomial p < 0.05;
    3. not clearly below the hand rule: on points where exactly one of the two
       hits, the rule is not better at two-sided sign-test p < 0.05.
  Object and button points are also reported separately. The 17 other games are
  held out from prompt tuning and scored once, after the prompt is frozen.
Writes results/coach/probe_<stamp>/{points.jsonl, answers.jsonl, report.md}.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # legacy/coach_track
ROOT = os.path.dirname(os.path.dirname(TRACK))                               # the repo
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))   # upgrades.py, gridtools.py
sys.path.insert(0, TRACK)                                  # the coach package
from coach.history import History, SUMMARY_VERSION, obj_key  # noqa: E402
from coach.advice import (BUTTON_NAMES, LLMClient, heuristic_advice, prompt_hash,  # noqa: E402
                          ranking)

MANIFEST = "results/confirm_upgrade/20260927_221915/manifest.tsv"
DEV_GAMES = ["tu93", "tr87", "dc22", "g50t", "vc33", "ft09", "m0r0", "cd82"]
STALL, AGAIN, PER_LEVEL = 1500, 3000, 4


# --------------------------------------------------------------------------------
# Building points (one recorded run at a time)
# --------------------------------------------------------------------------------
def transitions(corpus):
    for p in sorted(glob.glob(os.path.join(corpus, "shard_*.npz"))):
        with np.load(p) as z:
            f, a, nf, lv, an = z["frames"], z["actions"], z["next_frames"], z["levels"], z["action_nums"]
            for i in range(len(a)):
                yield f[i], int(a[i]), nf[i], int(lv[i]), int(an[i])


def winning_moves(corpus):
    out, prev = set(), None
    for p in sorted(glob.glob(os.path.join(corpus, "shard_*.npz"))):
        with np.load(p) as z:
            for lv, an in zip(z["levels"].tolist(), z["action_nums"].tolist()):
                if prev is not None and lv > prev[0]:
                    out.add(prev[1])
                prev = (lv, an)
    return out


def _target(h, frame, a, summary, by_label):
    """The scored target for move `a` made on `frame`, against `summary`."""
    if a < 5:
        names = {BUTTON_NAMES[b["action"]] for b in summary["buttons"]}
        nm = BUTTON_NAMES[a]
        return {"kind": "button", "button": nm} if nm in names else {"kind": "other", "why": "button not listed"}
    y, x = divmod(a - 5, 64)
    it = h.objects(frame).at(y, x)
    if it is None:
        return {"kind": "other", "why": "background click"}
    if by_label:
        ids = [o["id"] for o in summary["objects"] if o["label"] == it["label"]]
    else:
        ids = [o["id"] for o in summary["objects"] if (o["colour"], tuple(o["bbox"])) == obj_key(it)]
    return {"kind": "object", "id": ids[0]} if ids and summary["clicks"] else {"kind": "other", "why": "object not listed"}


def build_run(job):
    rundir, game, seed = job
    corpus = os.path.join(rundir, "transitions")
    wins = winning_moves(corpus)
    h = History()
    points, pending = [], []
    stuck = {}                                   # level -> [points made, next threshold]
    for f, a, nf, lv, an in transitions(corpus):
        if h.level is not None and lv == h.level.index:
            since = h.level.moves - h.level.last_new
            st = stuck.setdefault(lv, [0, STALL])
            if since >= st[1] and st[0] < PER_LEVEL:
                s = h.render(f)
                points.append({"game": game, "seed": seed, "kind": "stall", "level": lv, "action_num": an,
                               "since_new": since, "summary": s, "target": None})
                pending.append(len(points) - 1)
                st[0] += 1
                st[1] = since + AGAIN
        if an in wins:
            s = h.render(f)
            points.append({"game": game, "seed": seed, "kind": "prewin", "level": lv, "action_num": an,
                           "since_new": h.level.moves - h.level.last_new if h.level else 0,
                           "summary": s, "target": _target(h, f, a, s, by_label=True)})
        level_before = h.level.index if h.level is not None else None
        tgt_obj = h.objects(f).at(*divmod(a - 5, 64)) if a >= 5 else None
        h.observe(f, a, nf, lv)
        if level_before is not None and h.level.index != level_before:
            pending = []                             # level ended: the rest stay unresolved
            continue
        if h.level.last_new == h.level.moves:          # this move reached a new screen
            for i in pending:
                p = points[i]
                if a < 5:
                    p["target"] = _target(h, f, a, p["summary"], by_label=False)
                elif tgt_obj is None:
                    p["target"] = {"kind": "other", "why": "background click"}
                else:
                    ids = [o["id"] for o in p["summary"]["objects"]
                           if (o["colour"], tuple(o["bbox"])) == obj_key(tgt_obj)]
                    p["target"] = ({"kind": "object", "id": ids[0]} if ids and p["summary"]["clicks"]
                                   else {"kind": "other", "why": "object not listed"})
                p["target"]["after"] = an - p["action_num"]
            pending = []
            if lv in stuck:
                stuck[lv][1] = STALL
    for p in points:
        if p["target"] is None:
            p["target"] = {"kind": "other", "why": "unresolved"}
        for o in p["summary"]["objects"]:
            o["bbox"] = list(o["bbox"])
    return points


# --------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------
def hit(p, objs, aw, idx):
    """1 if the ranking hits the point's target, else 0 (button ties broken by
    a fixed draw per point, the same draw for every ranker)."""
    t = p["target"]
    if t["kind"] == "object":
        return int(t["id"] in objs[:3])
    if not aw:
        return 0
    top = max(aw.values())
    tied = sorted(k for k, v in aw.items() if v == top)
    return int(random.Random(1000 + idx).choice(tied) == t["button"])


def random_expectation(p):
    t = p["target"]
    if t["kind"] == "object":
        n = len(p["summary"]["objects"])
        return min(3, n) / n
    return 1 / len(p["summary"]["buttons"])


def poisson_binomial_sf(probs, k):
    """P(X >= k) for X = sum of independent Bernoulli(probs)."""
    dist = [1.0]
    for q in probs:
        nxt = [0.0] * (len(dist) + 1)
        for i, v in enumerate(dist):
            nxt[i] += v * (1 - q)
            nxt[i + 1] += v * q
        dist = nxt
    return sum(dist[k:]) if k < len(dist) else 0.0


def sign_test_two_sided(a, b):
    """Two-sided binomial test of a successes vs b, p = 0.5."""
    n = a + b
    if n == 0:
        return 1.0
    k = max(a, b)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def score(points, answers):
    """Rows per scored point with hits for each ranker."""
    rows = []
    for idx, p in enumerate(points):
        if p["target"]["kind"] not in ("object", "button"):
            continue
        s = p["summary"]
        h_objs, h_aw = ranking(heuristic_advice(s), s)
        row = {"idx": idx, "game": p["game"], "kind": p["kind"], "target": p["target"]["kind"],
               "random": random_expectation(p), "rule": hit(p, h_objs, h_aw, idx)}
        ans = answers.get(idx)
        if ans is not None:
            if ans.get("advice") is None:
                row["llm"] = 0
            else:
                l_objs, l_aw = ranking(ans["advice"], s)
                row["llm"] = hit(p, l_objs, l_aw, idx)
        rows.append(row)
    return rows


def verdict(rows):
    n = len(rows)
    exp = sum(r["random"] for r in rows)
    rule = sum(r["rule"] for r in rows)
    out = {"n": n, "random_expected": round(exp, 2), "rule_hits": rule}
    if rows and "llm" in rows[0]:
        llm = sum(r["llm"] for r in rows)
        p_rand = poisson_binomial_sf([r["random"] for r in rows], llm)
        only_llm = sum(1 for r in rows if r["llm"] and not r["rule"])
        only_rule = sum(1 for r in rows if r["rule"] and not r["llm"])
        p_rule = sign_test_two_sided(only_llm, only_rule)
        c1, c2 = n >= 15, (llm > exp and p_rand < 0.05)
        c3 = not (only_rule > only_llm and p_rule < 0.05)
        out.update(llm_hits=llm, p_llm_vs_random=p_rand, only_llm=only_llm, only_rule=only_rule,
                   p_llm_vs_rule=p_rule, c1=c1, c2=c2, c3=c3, go=c1 and c2 and c3)
    return out


# --------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--games", default=",".join(DEV_GAMES))
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8017/v1")
    ap.add_argument("--workers", type=int, default=4, help="concurrent LLM requests (server runs 4)")
    ap.add_argument("--jobs", type=int, default=12, help="processes for building points")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    games = a.games.split(",")
    out = a.out or os.path.join(ROOT, "results", "coach", "probe_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    jobs = []
    for line in open(os.path.join(ROOT, a.manifest)):
        rundir, game, seed, _ = line.rstrip("\n").split("\t")
        if game in games:
            jobs.append((os.path.join(ROOT, rundir) if not os.path.isabs(rundir) else rundir, game, int(seed)))
    t0 = time.time()
    with Pool(min(a.jobs, len(jobs))) as pool:
        points = [p for run in pool.map(build_run, jobs) for p in run]
    with open(os.path.join(out, "points.jsonl"), "w") as f:
        for p in points:
            f.write(json.dumps(p) + "\n")
    kinds = {}
    for p in points:
        k = (p["kind"], p["target"]["kind"] if p["target"]["kind"] != "other" else "not scored: " + p["target"]["why"])
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{len(points)} points from {len(jobs)} runs in {time.time() - t0:.0f} s")
    for k, v in sorted(kinds.items()):
        print(f"  {k[0]:7s} {k[1]:32s} {v}")

    answers = {}
    if not a.no_llm:
        client = LLMClient(a.url, cache_path=os.path.join(ROOT, "results", "advisor_cache", "probe.jsonl"))
        todo = [i for i, p in enumerate(points) if p["target"]["kind"] in ("object", "button")]
        t1 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            for i, rec in zip(todo, ex.map(lambda i: client.ask(points[i]["summary"], seed=i), todo)):
                answers[i] = rec
        with open(os.path.join(out, "answers.jsonl"), "w") as f:
            for i in todo:
                f.write(json.dumps(dict(answers[i], idx=i)) + "\n")
        print(f"{len(todo)} LLM answers in {time.time() - t1:.0f} s "
              f"({sum(1 for r in answers.values() if r['cached'])} from cache)")

    rows = score(points, answers)
    report(out, points, answers, rows, a, games)


def report(out, points, answers, rows, a, games):
    def table(sel, title):
        v = verdict(sel)
        lines = [f"**{title}**: {v['n']} scored points; random expects {v['random_expected']}; "
                 f"hand rule {v['rule_hits']}" + (f"; LLM {v['llm_hits']}" if "llm_hits" in v else "")]
        return lines, v
    lines = ["# Coach offline probe (step 4)", "",
             f"Games: {', '.join(games)}. Manifest: `{a.manifest}`. Summary `{SUMMARY_VERSION}`, "
             f"prompt `{prompt_hash()}`.", ""]
    allv = verdict(rows)
    if "go" in allv:
        lines += ["## G1 verdict (pre-registered in legacy/coach_track/tools/advice_probe.py)", "",
                  f"- 1. at least 15 scored points: {allv['n']} → {'PASS' if allv['c1'] else 'FAIL'}",
                  f"- 2. LLM beats random: {allv['llm_hits']} hits vs {allv['random_expected']} expected, "
                  f"p = {allv['p_llm_vs_random']:.4f} → {'PASS' if allv['c2'] else 'FAIL'}",
                  f"- 3. not clearly below the hand rule: LLM-only hits {allv['only_llm']}, rule-only "
                  f"{allv['only_rule']}, p = {allv['p_llm_vs_rule']:.4f} → {'PASS' if allv['c3'] else 'FAIL'}",
                  f"- **G1: {'GO' if allv['go'] else 'NO-GO'}**", ""]
    lines += ["## Hits by point type", "",
              "| points | n | random (expected) | hand rule | LLM |", "|---|---:|---:|---:|---:|"]
    for title, sel in [("all", rows),
                       ("object targets (top 3)", [r for r in rows if r["target"] == "object"]),
                       ("button targets (top 1)", [r for r in rows if r["target"] == "button"]),
                       ("pre-win", [r for r in rows if r["kind"] == "prewin"]),
                       ("stall", [r for r in rows if r["kind"] == "stall"])]:
        v = verdict(sel)
        lines.append(f"| {title} | {v['n']} | {v['random_expected']} | {v['rule_hits']} | {v.get('llm_hits', '-')} |")
    lines += ["", "## Per game", "", "| game | n | random | rule | LLM |", "|---|---:|---:|---:|---:|"]
    for g in games:
        v = verdict([r for r in rows if r["game"] == g])
        lines.append(f"| {g} | {v['n']} | {v['random_expected']} | {v['rule_hits']} | {v.get('llm_hits', '-')} |")
    unscored = {}
    for p in points:
        if p["target"]["kind"] == "other":
            unscored[p["target"]["why"]] = unscored.get(p["target"]["why"], 0) + 1
    lines += ["", f"Coverage: {len(rows)} of {len(points)} points scored; not scored: "
              + (", ".join(f"{k} {v}" for k, v in sorted(unscored.items())) or "none") + "."]
    if answers:
        recs = list(answers.values())
        lat = sorted(r["latency_s"] for r in recs)
        bad = sum(1 for r in recs if r.get("advice") is None)
        probs = sum(1 for r in recs if r.get("problems"))
        lines += ["", "## LLM cost and format", "",
                  f"- model `{recs[0]['model']}`; {len(recs)} answers; median latency {lat[len(lat) // 2]:.2f} s "
                  f"(4 concurrent); 90th percentile {lat[int(0.9 * (len(lat) - 1))]:.2f} s",
                  f"- unparseable answers: {bad}; answers with dropped ids or buttons: {probs}",
                  f"- median prompt {sorted(r['prompt_tokens'] for r in recs)[len(recs) // 2]} tokens, "
                  f"answer {sorted(r['completion_tokens'] for r in recs)[len(recs) // 2]} tokens"]
        rng = random.Random(0)
        pick = sorted(rng.sample(sorted(answers), min(20, len(answers))))
        lines += ["", "## 20 hypotheses for hand checking", "",
                  "| point | game | type | target | LLM top picks | hypothesis |", "|---|---|---|---|---|---|"]
        for i in pick:
            p, r = points[i], answers[i]
            t = p["target"]
            tdesc = f"object {t['id']}" if t["kind"] == "object" else t.get("button", "")
            if r.get("advice"):
                objs, aw = ranking(r["advice"], p["summary"])
                picks = f"objects {objs[:3]}" + (f"; buttons {aw}" if aw else "")
                hyp = r["advice"]["hypothesis"].replace("|", "/")
            else:
                picks, hyp = "-", "(unparseable)"
            lines.append(f"| {i} | {p['game']} | {p['kind']} | {tdesc} | {picks} | {hyp} |")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))
    print(f"\nwrote {out}/report.md")


if __name__ == "__main__":
    main()
