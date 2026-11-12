#!/usr/bin/env python3
"""gb_funnel.py - the funnel: what each Glassbox stage did on each game, side by side.

The problem it fixes: every stage writes its own numbers into its own files
(S1's summary, S2's llm.json, S3's book, S4's carry, S5's replay), so nobody
can see in one table where a game's cases are lost. The plan promises that
table for the report (section 6.1, "The funnel, one set of numbers per stage").

    uv run python tools/gb_funnel.py --runs g1,pool_g0b_g1,g3_gb \\
        [--replay cf_gb,cf_free] [--online results/glassbox/g4/online/manifest.tsv:GB] [--out funnel.md]

It only reads finished runs under results/glassbox/ (no LLM, no game, seconds).

  --runs    offline runs. Per game, on its training level: the changing cases,
            how many S1 describes as whole objects, how many lie in an action
            group S2 asked the LLM about, how many an admitted rule gets right,
            and how many the book gets partly and wholly right (S6's funnel,
            custom_agents/gb/s6_show.py). Then one line of each stage's own
            metrics: S2's answers, S3's rules and coverage, S4's T0 and T1 on
            the next level beside "nothing" and "memory".
  --replay  S5's replay runs (the offline counterfactual): avoidable waste and
            masking risk in the first actions of each new level.
  --online  the S5 arm of an online sweep, MANIFEST:ARM: what the guide carried,
            dropped, re-bound and demoted, and rule survival (the share of
            re-bound rules never demoted).

Where a run has S3's guard-fitter summary (G1, G3), the book and T1 are the ones
after the guards, as its gate read them (tools/gb_gate.py); otherwise S3's book
and S4's carry as written.
"""
import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from gb import artifacts as A  # noqa: E402
from gb import s6_show  # noqa: E402
from gb.core.evidence import LevelEvidence  # noqa: E402
from manifest import read_manifest  # noqa: E402

LLM_ORIGINS = ("first round", "refine", "residual", "retry")     # candidates that are LLM answers (gb_gate.py)


def _json(path):
    with open(path) as f:
        return json.load(f)


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def game_row(run_dir, game):
    """One game's numbers from one offline run (module doc)."""
    g = s6_show.load_game(run_dir, game)
    f = s6_show.funnel(g, LevelEvidence.load(g["train_path"]))
    rules = list(g["rules"].values())
    row = {"game": game, "level": f["level"], "changing": f["changing"], "s1": f["s1"], "called": f["in_called"],
           "admitted_cases": f["admitted"], "part": f["book_part"], "whole": f["book_whole"],
           "admitted": sum(r["v2_admitted"] for r in rules), "trusted": sum(r["v2_trusted"] for r in rules),
           "book_rules": f["book_rules"], "cell": f["cell_coverage"],
           "case": g["book"]["numbers"]["case_coverage"], "wrong": f["book_wrong"], "rescued": 0}
    carry = g["carry"]
    t0, t1 = (carry["t0"], carry["t1"]) if carry else (None, None)
    guards = glob.glob(os.path.join(run_dir, "s3_check", f"{game}_L*.guards.json"))
    if guards:                                          # the book and T1 after the guard fitter
        s = _json(guards[0])
        after = s["book"]["after"]
        row.update(book_rules=len(after["members"]), cell=after["cell_coverage"], case=after["case_coverage"],
                   wrong=after["wrong"], rescued=len(s["rescued"]), trusted=row["trusted"] + len(s["rescued"]))
        t1 = (s.get("t1") or {}).get("after") or t1
    if t1:
        rate = t1["per_move"]["rate"]
        row.update(next_level=g["k_next"], t0=t0["per_move"]["rate"]["book"] if t0 else None, t1=rate["book"],
                   nothing=rate["nothing"], memory=rate["memory"], t1_wrong=t1["wrong"],
                   beats_both=t1["per_move"]["beats_both"],
                   carried=len(t1["members"]) if "members" in t1 else t1.get("rules_in"))
    llm = g["llm"] or {}
    if "calls" in llm:                                  # S2 asked the LLM in this run
        answers = [c for c in g["code"].values() if c.get("origin") in LLM_ORIGINS]
        checked = [c for c in answers if c.get("stage") == "checked"]
        row.update(calls=llm["calls"], answers=len(answers), compiled=len(checked), ran_out=llm["ran_out"],
                   tokens=llm["tokens"], with_h1=sum(bool(c.get("primitives")) for c in checked))
    return row


def games_of(run_dir):
    return s6_show.games_in(run_dir)


def pct(x, digits=1):
    return "-" if x is None else f"{x:.{digits}%}"


def share(n, d):
    return "-" if not d else f"{n / d:.0%}"


def offline_tables(name, rows):
    """Two tables for one run: the funnel in cases, and each stage's own metrics."""
    n = len(rows)
    tot = lambda k: sum(r.get(k) or 0 for r in rows)
    lines = [f"### `{name}` ({n} games)", "",
             "Training level. Each share is of the level's distinct changing cases.", "",
             "| Game (level) | Changing cases | S1 describes it | In a group S2 asked about | An admitted rule gets it "
             "| Book: part right | Book: whole screen |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c = r["changing"]
        lines.append(f"| {r['game']} (L{r['level']}) | {c:,} | {share(r['s1'], c) if r['s1'] is not None else '-'} | "
                     f"{share(r['called'], c)} | {share(r['admitted_cases'], c)} | {share(r['part'], c)} | "
                     f"{share(r['whole'], c)} |")
    # the mean over games, not the pool: one game can hold most of the cases (tr87: 40,000)
    mean_share = lambda k: (f"{st.mean(r[k] / r['changing'] for r in rows if r['changing']):.0%}"
                            if all(r[k] is not None for r in rows) else "-")
    lines.append(f"| **mean over games** | {tot('changing'):,} in all | {mean_share('s1')} | {mean_share('called')} | "
                 f"{mean_share('admitted_cases')} | {mean_share('part')} | {mean_share('whole')} |")
    asked = [r for r in rows if "answers" in r]
    lines += ["",
              "| Game | S2: answers | compiled | out of room | tokens / answer | use H1 | S3: admitted | trusted "
              "| in book | cell coverage | wrong | S4: T0 | T1 | nothing | memory | T1 wrong |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        s2 = (f"{r['answers']} | {share(r['compiled'], r['answers'])} | {share(r['ran_out'], r['answers'])} | "
              f"{r['tokens'] / max(r['answers'], 1):,.0f} | {share(r['with_h1'], r['compiled'])}"
              if "answers" in r else "- | - | - | - | -")
        s4 = (f"{pct(r.get('t0'))} | {pct(r['t1'])} | {pct(r['nothing'])} | {pct(r['memory'])} | {pct(r['t1_wrong'])}"
              if "t1" in r else "- | - | - | - | -")
        lines.append(f"| {r['game']} | {s2} | {r['admitted']} | {r['trusted']} | {r['book_rules']} | {pct(r['cell'])} | "
                     f"{pct(r['wrong'])} | {s4} |")
    if asked:
        a = sum(r["answers"] for r in asked)
        s2 = (f"{a:,} | {share(sum(r['compiled'] for r in asked), a)} | {share(sum(r['ran_out'] for r in asked), a)} | "
              f"{sum(r['tokens'] for r in asked) / a:,.0f} | "
              f"{share(sum(r['with_h1'] for r in asked), sum(r['compiled'] for r in asked))}")
    else:
        s2 = "- | - | - | - | -"
    carried = [r for r in rows if "t1" in r]
    mean = lambda k: pct(st.mean(r[k] for r in carried if r.get(k) is not None)) if carried else "-"
    lines.append(f"| **all** | {s2} | {tot('admitted')} | {tot('trusted')} | {tot('book_rules')} | "
                 f"{pct(st.mean(r['cell'] for r in rows))} (mean) | {pct(st.mean(r['wrong'] for r in rows))} (mean) | "
                 f"{mean('t0')} | {mean('t1')} | {mean('nothing')} | {mean('memory')} | {mean('t1_wrong')} |")
    beats = [r["game"] for r in carried if r.get("beats_both")]
    lines += ["",
              f"Games with a next level to carry to: {len(carried)} of {n}. T1 beats both \"nothing\" and \"memory\" "
              f"(both intervals above zero) on: {', '.join(beats) or 'none'}. Rules made exact by a fitted guard: "
              f"{tot('rescued')}."]
    return lines


def replay_table(names):
    """S5's replay runs side by side: avoidable waste and masking risk per game."""
    per = {n: _json(os.path.join(A.results_dir(), n, "s5_act", "replay_summary.json")) for n in names}
    games = [g for g in per[names[0]]]
    lines = ["| Game | " + " | ".join(f"`{n}`: avoidable waste | masking risk | covered by" for n in names) + " |",
             "|---|" + "---:|---:|---|" * len(names)]
    for g in games:
        cells = []
        for n in names:
            r = per[n][g]
            by = ", ".join(f"{k} {v:,}" for k, v in sorted(r["by"].items())) or "-"
            cells.append(f"{r['waste']:.1%} of {r['window']:,} | {r['masking']:.2%} | {by}")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    return lines


def online_table(spec):
    """What S5 did online, per game, from each run's glassbox/events.jsonl and s5_stats.json."""
    path, arm = spec.rsplit(":", 1)
    per = collections.defaultdict(collections.Counter)
    for r in read_manifest(path, strict=True):
        if r["arm"] != arm:
            continue
        d = os.path.join(r["run_dir"] if os.path.isabs(r["run_dir"]) else os.path.join(ROOT, r["run_dir"]), "glassbox")
        c = per[r["game"]]
        for e in _jsonl(os.path.join(d, "events.jsonl")):
            c[e["kind"]] += 1
        s = _json(os.path.join(d, "s5_stats.json"))
        c["moves"] += s.get("moves", 0)
        c["changed"] += s.get("changed_move", 0)
        c["runs"] += 1
    lines = ["| Game | Runs | Moves the guide changed | Colours carried | dropped | Rules re-bound | demoted "
             "| Rule survival | Goal hints set | dropped |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tot = collections.Counter()
    for g, c in per.items():
        tot.update(c)
        lines.append(_online_row(g, c))
    lines.append(_online_row("**all**", tot))
    return lines


def _online_row(name, c):
    bound, demoted = c["rule re-bound"], c["rule demoted"]
    return (f"| {name} | {c['runs']} | {share(c['changed'], c['moves'])} | {c['colour no-op carried']} | "
            f"{c['colour no-op dropped']} | {bound} | {demoted} | {share(bound - demoted, bound)} | "
            f"{c['goal hints']} | {c['goal hint dropped']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="", help="offline runs under results/glassbox/, comma-separated")
    ap.add_argument("--replay", default="", help="S5 replay runs, comma-separated (the first names the games)")
    ap.add_argument("--online", default="", help="MANIFEST:ARM of an online sweep's S5 arm")
    ap.add_argument("--out", default="", help="optional .md path")
    a = ap.parse_args()
    lines = []
    for name in [n for n in a.runs.split(",") if n]:
        run_dir = os.path.join(A.results_dir(), name)
        lines += offline_tables(name, [game_row(run_dir, g) for g in games_of(run_dir)]) + [""]
    if a.replay:
        lines += ["### S5 replay: " + ", ".join(f"`{n}`" for n in a.replay.split(",")), "",
                  "Levels 2 and up, the first actions of each level (`[act] waste_window`), pooled over seeds.", ""]
        lines += replay_table(a.replay.split(",")) + [""]
    if a.online:
        lines += [f"### S5 online: `{a.online}`", ""] + online_table(a.online) + [""]
    if not lines:
        ap.error("give --runs, --replay or --online")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write("# The Glassbox funnel\n\n" + text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
