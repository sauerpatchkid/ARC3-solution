#!/usr/bin/env python3
"""wm_thinkcap.py - what is lost if the rule writer's thinking stops early?
(docs/plans/rulebook-v2.md section 10.5; for CP2, whose answers are capped at
12,288 tokens where Tier 0a and CP1 allow 20,480.) Dev and seen games only.

Takes recorded first-round answers whose prompts can be rebuilt exactly: Tier
0a's round 1 (8 dev games, v1's rule format) and the CP1 smoke test's round 1
(lp85, v2's format). For a cap of N tokens:
  - an answer that ended within N tokens is unchanged, and keeps its outcome;
  - a longer one is cut where its thinking reaches N tokens (what a cap of N
    would have left), and the code-only retry is asked for the code, exactly as
    v1 does for an answer that ran out of room (thinking off, 4,096 tokens, seed
    51, the retry shown the last 24,000 characters of the reasoning);
  - the resulting rule is checked on the same evidence as the original, and
    the two outcomes are compared: code that runs, admitted, exact (v1's
    plan-eligible; v2: trusted).
Only the retries are new answers. Token counts of the cut answers come from
their length in characters at a rate measured on CP0's answers (one answer per
request there, so its token counts are exact).

  experiments/rulebook/llm_run.sh start thinkcap            # ~1 h of GPU
  uv run python tools/wm_thinkcap.py --dry-run               # no server: how many answers each cap cuts
Writes <out>/{report.md, results.json, llm_cache.jsonl}.
"""
import argparse
import collections
import json
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # legacy/rulebook
ROOT = os.path.dirname(os.path.dirname(TRACK))                        # the repo
sys.path.insert(0, TRACK)                                             # the frozen wm package
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))
sys.path.insert(0, os.path.join(TRACK, "tools"))
from wm import rules as R  # noqa: E402
from wm import rules2 as R2  # noqa: E402
from wm import tiers  # noqa: E402
from wm.check import check_rule, grade  # noqa: E402
from wm.evidence import LevelEvidence  # noqa: E402
from wm.llm import ChatClient  # noqa: E402

V1 = os.path.join(ROOT, "results", "rulebook", "tier0a")
CP0 = os.path.join(ROOT, "results", "rulebook", "v2", "nearmiss", "llm_cache.jsonl")
SMOKE = os.path.join(ROOT, "results", "rulebook", "v2", "tier0a_v2_smoke")
CAPS = (12288, 6144)
WORKERS = 6


def rng_for(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def chars_per_token():
    """Characters per generated token, from CP0's answers (n = 1, exact counts)."""
    chars = tokens = 0
    for line in open(CP0):
        for o in json.loads(line)["outputs"]:
            chars += len(o["text"])
            tokens += o["tokens"]
    return chars / tokens


def reader(cache_path):
    """A client that only answers from an existing cache."""
    c = ChatClient(cache_path=cache_path)
    c._served = next(iter({r["model"] for r in c.cache.values()}))

    def refuse(messages, n):
        raise LookupError("not in the cache: the prompt was not rebuilt exactly")
    c.mock = refuse
    c.cache_path = None                                        # never written to
    return c


def tier0a_answers():
    """Tier 0a's round-1 answers with their conversations and outcomes."""
    client, out = reader(os.path.join(V1, "llm_cache.jsonl")), []
    for game in tiers.DEV:
        with open(os.path.join(V1, f"{game}.json")) as f:
            v1 = json.load(f)
        ev_path = os.path.join(V1, f"{game}_train.npz")
        ev = LevelEvidence.load(ev_path)
        cands = [c for c in v1["candidates"] if c["round"] == 1]
        by_code = {c["code"]: c for c in v1["candidates"] if c.get("code") and c["stage"] == "checked"}
        for g in ev.groups[:12]:
            conv = R.conversation(R.group_prompt(ev, g, rng_for(game, g, "prompt")), thinking=True)
            outs = client.complete(conv, n=4, max_tokens=R.MAX_TOKENS[True], thinking=True, seed=1, **R.SAMPLING[True])
            for k, o in enumerate(outs):
                mine = [c for c in cands if c["group"] == g and c["k"] == k]
                final = next((c for c in mine if c.get("repaired")), mine[0] if mine else None)
                if final is not None and final["stage"] == "duplicate":
                    final = by_code.get(final["code"], final)
                out.append({"source": "tier0a", "game": game, "group": g, "k": k, "conv": conv, "text": o["text"],
                            "finished": o["finish"] == "stop", "ev_path": ev_path, "format": 1,
                            "original": outcome_v1(final)})
    return out


def smoke_answers():
    """The CP1 smoke test's round-1 answers (v2's rule format)."""
    import wm_tier0a_v2 as T
    client, out = reader(os.path.join(SMOKE, "llm_cache.jsonl")), []
    with open(os.path.join(SMOKE, "lp85.json")) as f:
        run = json.load(f)
    ev_path = os.path.join(SMOKE, "lp85_train.npz")
    ev = LevelEvidence.load(ev_path)
    for g in [r["group"] for r in run["groups"]]:
        parts, instruction = R2.first_round(ev, g, rng_for("lp85", g, "prompt"))
        conv = R2.conversation(parts, instruction)
        outs = client.complete(conv, n=4, max_tokens=R.MAX_TOKENS[True], thinking=True, seed=1, **R.SAMPLING[True])
        for k, o in enumerate(outs):
            final = next(c for c in run["candidates"] if c["group"] == g and c["origin"] == "first round" and c["k"] == k)
            out.append({"source": "smoke", "game": "lp85", "group": g, "k": k, "conv": conv, "text": o["text"],
                        "finished": o["finish"] == "stop", "ev_path": ev_path, "format": 2,
                        "original": outcome_v2(final), "T": T})
    return out


def outcome_v1(c):
    if c is None or c["stage"] != "checked":
        return {"runs": False, "admitted": False, "exact": False}
    g = c["grade"]
    return {"runs": True, "admitted": bool(g["admitted"]), "exact": bool(g["plan_eligible"])}


def outcome_v2(c):
    if c["stage"] != "checked":
        return {"runs": False, "admitted": False, "exact": False}
    t = c["trust"]
    return {"runs": True, "admitted": bool(t["admitted"]), "exact": bool(t["trusted"])}


def cut(text, cap, rate):
    """(the answer as a cap of `cap` tokens would have left it, or None if it fits)."""
    if len(text) <= cap * rate:
        return None
    thinking = text.split("</think>")[0]
    return thinking[:int(cap * rate)]


def retry(client, a, partial):
    """The code-only retry on a cut answer, and its outcome."""
    finish = R.FINISH if a["format"] == 1 else R2.FINISH
    o = client.complete(a["conv"] + [{"role": "assistant", "content": R.tail(partial)},
                                     {"role": "user", "content": finish}],
                        n=1, max_tokens=R.MAX_TOKENS[False], thinking=False, seed=51, **R.SAMPLING[False])[0]
    code = R.extract_code(o["text"])
    if code is None:
        return {"runs": False, "admitted": False, "exact": False}, o["tokens"]
    if a["format"] == 1:
        res = check_rule(code, a["ev_path"])
        if res["stage"] != "checked":
            return {"runs": False, "admitted": False, "exact": False}, o["tokens"]
        g = grade(res, LevelEvidence.load(a["ev_path"]), a["group"])
        return {"runs": True, "admitted": bool(g["admitted"]), "exact": bool(g["plan_eligible"])}, o["tokens"]
    arm = a["T"].evaluate(code, a["group"], LevelEvidence.load(a["ev_path"]), a["ev_path"])
    if arm["stage"] != "checked":
        return {"runs": False, "admitted": False, "exact": False}, o["tokens"]
    t = arm["trust"]
    return {"runs": True, "admitted": bool(t["admitted"]), "exact": bool(t["trusted"])}, o["tokens"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rulebook", "v2", "thinkcap"))
    ap.add_argument("--url", default="http://127.0.0.1:8018/v1")
    ap.add_argument("--dry-run", action="store_true", help="no server: rebuild the prompts and count what each cap cuts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rate = chars_per_token()
    answers = tier0a_answers() + smoke_answers()
    print(f"{len(answers)} answers rebuilt from their caches; {rate:.2f} characters per token", flush=True)
    for cap in CAPS:
        n = sum(cut(x["text"], cap, rate) is not None for x in answers)
        print(f"cap {cap:,}: {n} of {len(answers)} answers would be cut", flush=True)
    if a.dry_run:
        return
    client = ChatClient(a.url, cache_path=os.path.join(a.out, "llm_cache.jsonl"))
    print(f"model: {client.served_model()}", flush=True)
    rows = []
    for cap in CAPS:
        todo = [(x, p) for x in answers if (p := cut(x["text"], cap, rate)) is not None]
        with ThreadPoolExecutor(WORKERS) as ex:
            got = list(ex.map(lambda xp: retry(client, *xp), todo))
        new = {id(x): g for (x, _), g in zip(todo, got)}
        for x in answers:
            o, toks = new.get(id(x), (x["original"], None))
            full = len(x["text"]) / rate
            rows.append({"cap": cap, "source": x["source"], "game": x["game"], "group": x["group"], "k": x["k"],
                         "cut": id(x) in new, "original": x["original"], "capped": o,
                         "tokens_original": round(full), "tokens_capped": round(min(full, cap) + (toks or 0))})
        print(f"cap {cap:,}: {len(todo)} retries done", flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"chars_per_token": rate, "rows": rows}, f, indent=1)
    report(a.out, rows, client)


def report(out, rows, client):
    m = client.meter
    L = ["# What is lost if the rule writer's thinking stops early", "",
         "tools/wm_thinkcap.py; plan section 10.5. Recorded first-round answers (Tier 0a, 8 dev games, v1's "
         "format; the CP1 smoke test, lp85, v2's format). An answer longer than the cap is cut where its thinking "
         "reaches the cap and finished by v1's code-only retry; shorter answers keep their outcome. Exact = v1's "
         "plan-eligible, or v2's trusted.",
         f"Model `{client.served_model()}`. Retries: {m['completions']} new answers, {m['tokens']:,} tokens, "
         f"{m['seconds'] / 60:.0f} request-minutes.", "",
         "| cap | source | answers | cut | runs: full / capped | admitted: full / capped | exact: full / capped | "
         "tokens: full / capped |", "|---:|---|---:|---:|---|---|---|---|"]
    for (cap, src), rs in sorted(collections.defaultdict(list, {k: v for k, v in _groups(rows).items()}).items(),
                                 key=lambda kv: (-kv[0][0], kv[0][1])):
        cnt = lambda when, what: sum(r[when][what] for r in rs)
        L.append(f"| {cap:,} | {src} | {len(rs)} | {sum(r['cut'] for r in rs)} | {cnt('original', 'runs')} / "
                 f"{cnt('capped', 'runs')} | {cnt('original', 'admitted')} / {cnt('capped', 'admitted')} | "
                 f"{cnt('original', 'exact')} / {cnt('capped', 'exact')} | "
                 f"{sum(r['tokens_original'] for r in rs):,} / {sum(r['tokens_capped'] for r in rs):,} |")
    L += ["", "Among the CUT answers only:", "",
          "| cap | cut answers | admitted: full / capped | exact: full / capped | lost an admitted rule | gained one |",
          "|---:|---:|---|---|---:|---:|"]
    for cap in CAPS:
        rs = [r for r in rows if r["cap"] == cap and r["cut"]]
        cnt = lambda when, what: sum(r[when][what] for r in rs)
        lost = sum(r["original"]["admitted"] and not r["capped"]["admitted"] for r in rs)
        gained = sum(r["capped"]["admitted"] and not r["original"]["admitted"] for r in rs)
        L.append(f"| {cap:,} | {len(rs)} | {cnt('original', 'admitted')} / {cnt('capped', 'admitted')} | "
                 f"{cnt('original', 'exact')} / {cnt('capped', 'exact')} | {lost} | {gained} |")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def _groups(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["cap"], r["source"])].append(r)
    return g


if __name__ == "__main__":
    main()
