#!/usr/bin/env python3
"""gb_gate.py - a gate's verdict from a finished run, as its pre-registration section defines it.

    uv run python tools/gb_gate.py g1 --run g1 --base g0b_bars
    uv run python tools/gb_gate.py g3 --run g3_gbt --base g3_gb

G1 (docs/3-glassbox/glassbox-prereg.md section 4). For each of the 8 dev games,
from the run's guard-fitter summary (s3_check/<game>_L<k>.guards.json, "after":
S3's book from S2's candidates plus every narrowed rule the fitter rescued):
  primary  the book's cell coverage on the training level is HIGHER than the
           base run's (G0b: CP1's rules on the same evidence), on >= 5 of 8 games
  safety   T1 is wrong on <= 5% of the transfer level's changing cases, on
           every game
  PASS     primary and safety.
Reported beside it, not gating: CP1's own criterion (T1 beats "nothing" and
"memory" with both intervals above zero), case coverage, trusted rules, the
LLM's calls, tokens and answers that ran out of room, H1 use, guard rescues.
Writes <run>/g1_verdict.json and <run>/g1_verdict.md; exit status 0 either way.

G3 (pre-registration section 6). On the 8 A/B-offline games, GB+T (--run: S2 started
from H3's adopted templates) against GB (--base: the same stages, no templates), each
game's numbers read as G1's are:
  primary  GB+T has MORE trusted rules AND a HIGHER training-level cell coverage than GB,
           on >= 5 of 8 games
  wrong    T1's wrong cases, pooled over the games with a transfer level (each game's
           wrong share x its changing cases, so to about one case), no higher than GB's
  overrun  answers that ran out of room, pooled over the LLM's answers (templates and
           guard-narrowed rules are not answers), no higher than GB's
  PASS     all three.
Reported: templates adopted, in the book and carried (reuse), LLM requests saved,
negative transfer (a game where GB+T's coverage or T1 is below GB's).
Writes <run>/g3_verdict.json and <run>/g3_verdict.md.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

import tiers  # noqa: E402
from gb import artifacts as A  # noqa: E402

PRIMARY_GAMES, SAFETY_WRONG = 5, 0.05
LLM_ORIGINS = ("first round", "refine", "residual", "retry")     # candidates that are LLM answers


def summary(run_dir, game):
    p = glob.glob(os.path.join(run_dir, "s3_check", f"{game}_L*.guards.json"))
    if not p:
        raise FileNotFoundError(f"{run_dir}: no guard-fitter summary for {game}")
    with open(p[0]) as f:
        return json.load(f)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def game_numbers(run_dir, game):
    s = summary(run_dir, game)
    book, t1 = s["book"]["after"], (s.get("t1") or {}).get("after")
    rules = jsonl(os.path.join(run_dir, "s3_check", game, "rules.jsonl"))
    out = {"cell_coverage": book["cell_coverage"], "case_coverage": book["case_coverage"],
           "book_rules": len(book["members"]), "training_wrong": book["wrong"],
           "trusted": sum(r["v2_trusted"] for r in rules) + len(s["rescued"]),
           "admitted": sum(r["v2_admitted"] for r in rules), "guard_rescues_after": len(s["rescued"])}
    if t1:
        pm = t1["per_move"]
        out.update(t1_rate=pm["rate"]["book"], t1_nothing=pm["rate"]["nothing"], t1_memory=pm["rate"]["memory"],
                   t1_wrong=t1["wrong"], t1_vs_nothing=pm["diff"]["nothing"], t1_vs_memory=pm["diff"]["memory"],
                   beats_both=pm["beats_both"], t1_rules=len(t1["members"]), t1_changing=t1["changing_cases"],
                   t1_wrong_cases=round(t1["wrong"] * t1["changing_cases"]))
    llm = os.path.join(run_dir, "s2_write", game, "llm.json")
    m = json.load(open(llm)) if os.path.exists(llm) else {}
    if "calls" in m:                                            # written by S2 (an imported run's is CP1's)
        cands = jsonl(os.path.join(run_dir, "s2_write", game, "candidates.jsonl"))
        checked = [c for c in cands if c.get("stage") == "checked"]
        trusted = [c for c in checked if c["trust"]["trusted"]]
        out.update(llm_calls=m["calls"], candidates=m["candidates"], tokens=m["tokens"], ran_out=m["ran_out"],
                   guard_rescues_in_loop=sum(c.get("origin") == "guard" and c["trust"]["trusted"] for c in checked),
                   with_primitives=round(sum(bool(c.get("primitives")) for c in checked) / len(checked), 3)
                   if checked else 0.0,
                   trusted_with_primitives=sum(bool(c.get("primitives")) for c in trusted),
                   failed_groups=[g["group"] for g in m["groups"] if g.get("error")],
                   answers=sum(c.get("origin") in LLM_ORIGINS for c in cands), requests=m.get("requests"))
    return out


def g1(run, base):
    games = list(tiers.TIERS["dev"])
    rows = {g: {"g1": game_numbers(run.dir, g), "base": game_numbers(base.dir, g)} for g in games}
    higher = [g for g in games if rows[g]["g1"]["cell_coverage"] > rows[g]["base"]["cell_coverage"]]
    unsafe = [g for g in games if rows[g]["g1"].get("t1_wrong", 0.0) > SAFETY_WRONG]
    primary, safety = len(higher) >= PRIMARY_GAMES, not unsafe
    return {"gate": "G1", "run": run.name, "base": base.name, "games": rows, "higher": higher,
            "primary": primary, "unsafe": unsafe, "safety": safety, "pass": primary and safety,
            "beats_both": [g for g in games if rows[g]["g1"].get("beats_both")],
            "failed_groups": {g: rows[g]["g1"].get("failed_groups") for g in games if rows[g]["g1"].get("failed_groups")}}


def template_numbers(run_dir, game):
    """What H3 did on one game of a GB+T run: tried, adopted, in S2's candidates, trusted, in the
    book, carried by S4."""
    p = os.path.join(run_dir, "h3_templates", f"{game}.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        h3 = json.load(f)
    cands = jsonl(os.path.join(run_dir, "s2_write", game, "candidates.jsonl"))
    tmpl = {c["id"]: c for c in cands if c.get("origin") == "template"}
    books = glob.glob(os.path.join(run_dir, "s3_check", f"{game}_L*.book.json"))
    members = set(json.load(open(books[0]))["members"]) if books else set()
    carried = {}
    for pc in glob.glob(os.path.join(run_dir, "s4_carry", f"{game}_L*.carry.json")):
        carried.update({r["id"]: r for r in json.load(open(pc))["rules"]})
    in_book = [i for i in tmpl if i in members]
    return {"tried": h3["tried"], "adopted": len(h3["adopted"]),
            "adopted_templates": sorted({r["tid"] for r in h3["adopted"]}),
            "adopted_groups": sorted({r["group"] for r in h3["adopted"]}),
            "trusted": sum(c["stage"] == "checked" and c["trust"]["trusted"] for c in tmpl.values()),
            "in_book": len(in_book), "in_book_templates": sorted({tmpl[i]["template"] for i in in_book}),
            "carried": sum(bool(carried.get(i, {}).get("in_t1")) for i in in_book),
            "not_carried": {tmpl[i]["template"]: carried[i].get("flag") or carried[i].get("why")
                            for i in in_book if i in carried and not carried[i].get("in_t1")}}


def g3(run, base):
    games = list(tiers.TIERS["ab_offline"])
    rows = {g: {"gbt": game_numbers(run.dir, g), "gb": game_numbers(base.dir, g),
                "templates": template_numbers(run.dir, g)} for g in games}
    better = [g for g in games if rows[g]["gbt"]["trusted"] > rows[g]["gb"]["trusted"]
              and rows[g]["gbt"]["cell_coverage"] > rows[g]["gb"]["cell_coverage"]]

    def pooled(arm, num, den):
        n, d = (sum(rows[g][arm].get(k) or 0 for g in games) for k in (num, den))
        return {"count": n, "of": d, "share": round(n / d, 4) if d else 0.0}
    wrong = {arm: pooled(arm, "t1_wrong_cases", "t1_changing") for arm in ("gbt", "gb")}
    overrun = {arm: pooled(arm, "ran_out", "answers") for arm in ("gbt", "gb")}
    primary = len(better) >= PRIMARY_GAMES
    wrong_ok = wrong["gbt"]["share"] <= wrong["gb"]["share"]
    overrun_ok = overrun["gbt"]["share"] <= overrun["gb"]["share"]
    negative = [g for g in games if rows[g]["gbt"]["cell_coverage"] < rows[g]["gb"]["cell_coverage"]
                or rows[g]["gbt"].get("t1_rate", 0) < rows[g]["gb"].get("t1_rate", 0)]
    saved = {g: (rows[g]["gb"].get("requests") or 0) - (rows[g]["gbt"].get("requests") or 0) for g in games}
    return {"gate": "G3", "run": run.name, "base": base.name, "games": rows, "better": better, "primary": primary,
            "wrong": wrong, "wrong_ok": wrong_ok, "overrun": overrun, "overrun_ok": overrun_ok,
            "pass": primary and wrong_ok and overrun_ok, "negative_transfer": negative, "requests_saved": saved,
            "failed_groups": {g: rows[g][arm].get("failed_groups") for g in games for arm in ("gbt", "gb")
                              if rows[g][arm].get("failed_groups")}}


def g3_table(v):
    w, o = v["wrong"], v["overrun"]
    lines = [f"# G3 verdict: {'PASS' if v['pass'] else 'FAIL'}", "",
             f"Primary (more trusted rules and higher cell coverage than {v['base']} on >= {PRIMARY_GAMES} of 8): "
             f"{len(v['better'])} of 8 ({', '.join(v['better']) or 'none'}): {'met' if v['primary'] else 'not met'}.  "
             f"T1 wrong, pooled: {w['gbt']['share']:.2%} vs {w['gb']['share']:.2%} "
             f"({'met' if v['wrong_ok'] else 'NOT met'}).  Overrun, pooled: {o['gbt']['share']:.1%} vs "
             f"{o['gb']['share']:.1%} ({'met' if v['overrun_ok'] else 'NOT met'}).", "",
             "| Game | Cell coverage (GB -> GB+T) | Trusted | Templates tried / adopted / in book / carried | "
             "T1 per move (GB -> GB+T) | Requests (GB -> GB+T) |",
             "|---|---|---|---|---|---|"]
    for g, r in v["games"].items():
        a, b, t = r["gbt"], r["gb"], r["templates"]
        t1 = (f"{b['t1_rate']:.1%} -> {a['t1_rate']:.1%}" if "t1_rate" in a and "t1_rate" in b else "-")
        lines.append(f"| {g} | {b['cell_coverage']:.1%} -> {a['cell_coverage']:.1%} | {b['trusted']} -> {a['trusted']} | "
                     f"{t.get('tried', 0)} / {t.get('adopted', 0)} / {t.get('in_book', 0)} / {t.get('carried', 0)} | "
                     f"{t1} | {b.get('requests', '-')} -> {a.get('requests', '-')} |")
    lines += ["", f"Negative transfer (coverage or T1 below GB's): {', '.join(v['negative_transfer']) or 'none'}."]
    if v["failed_groups"]:
        lines += ["", f"Groups that failed: {v['failed_groups']}"]
    return "\n".join(lines) + "\n"


def table(v):
    lines = [f"# G1 verdict: {'PASS' if v['pass'] else 'FAIL'}",
             "",
             f"Primary (cell coverage above {v['base']}'s on >= {PRIMARY_GAMES} of 8): {len(v['higher'])} of 8 "
             f"({', '.join(v['higher']) or 'none'}): {'met' if v['primary'] else 'not met'}.  "
             f"Safety (T1 wrong <= {SAFETY_WRONG:.0%} on every game): "
             + ("met" if v["safety"] else f"NOT met on {', '.join(v['unsafe'])}") + ".",
             "",
             "| Game | Cell coverage (base -> G1) | Case coverage | Trusted | T1 per move (nothing, memory) | T1 wrong | "
             "Beats both | Calls | Tokens |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g, r in v["games"].items():
        a, b = r["g1"], r["base"]
        t1 = (f"{a['t1_rate']:.1%} ({a['t1_nothing']:.1%}, {a['t1_memory']:.1%})" if "t1_rate" in a else "-")
        lines.append(f"| {g} | {b['cell_coverage']:.1%} -> {a['cell_coverage']:.1%} | {b['case_coverage']:.1%} -> "
                     f"{a['case_coverage']:.1%} | {b['trusted']} -> {a['trusted']} | {t1} | "
                     f"{a.get('t1_wrong', 0):.1%} | {'yes' if a.get('beats_both') else 'no'} | "
                     f"{a.get('llm_calls', '-')} | {a.get('tokens', 0):,} |")
    if v["failed_groups"]:
        lines += ["", f"Groups that failed: {v['failed_groups']}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gate", choices=["g1", "g3"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    root = A.results_dir()
    run, base = (type("R", (), {"name": n, "dir": os.path.join(root, n)}) for n in (a.run, a.base))
    v = g1(run, base) if a.gate == "g1" else g3(run, base)
    text = table(v) if a.gate == "g1" else g3_table(v)
    with open(os.path.join(run.dir, f"{a.gate}_verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    with open(os.path.join(run.dir, f"{a.gate}_verdict.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
