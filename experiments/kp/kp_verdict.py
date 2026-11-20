#!/usr/bin/env python3
"""kp_verdict.py - the rule of the Knowledge Planner's test, and the tables read beside it.

The problem it fixes: the test's rule was written before its runs
(docs/4-knowledge-planner/kp-test.md). This applies it the same way every
time, from the sweep's manifest, and gives no verdict before every run is in.

    uv run python experiments/kp/kp_verdict.py                  the test: the verdict and the tables
    DEV=1 uv run python experiments/kp/kp_verdict.py            the dev sweep: the tables only (it has no rule)
    ... [--sweep results/kp/test/<stamp>] [--partial]           another sweep; tables for a sweep not yet complete

It writes <sweep>/verdict.md and <sweep>/runs.csv (one row per run: results/
is not in git) and prints the short version.

THE RULE, as the test page has it:
  score             an arm's levels completed by 5,000 actions, summed over its runs
                    (the engine's count, as compute_metrics.py reports it)
  clears the noise  SOT-KP's score is above both SOT's and SOT2's, AND SOT-KP - SOT is above
                    the 97.5th percentile of the chance gap
  the chance gap    how far two totals of the SAME agent come apart: this test's own
                    differences SOT2 - SOT, one per game and seed, drawn again with
                    replacement, each with a sign at random, and summed; 10,000 times, seed 0
  otherwise         the planner is inside the noise at this budget, whatever its total

Everything else here is reported and is not part of the rule: by tier, the
click games apart from the games with no click, levels by 1,000 and 2,000
actions, the interval with games resampled (upgrade screen 3's), pairs better
and worse, level races, what the planner did, the funnel, run times.
"""
import argparse
import csv
import json
import os
import random
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "custom_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kp_offline  # noqa: E402  (the funnel's steps)
import kp_test as T  # noqa: E402
import levels_by_budget as L  # noqa: E402  (level races; and paired_compare's bootstrap)
import manifest as _manifest  # noqa: E402
import tiers  # noqa: E402

BUDGETS = (1000, 2000, 5000)
ANCHOR = (28, 66)           # a baseline arm outside this many levels on 50 runs: check the set-up
SCREEN3 = "SOT 22, the adopted Goose 15 and 15 (25 games, seed 0, 5,000 actions)"
TIER_GROUPS = (("dev", tiers.DEV), ("seen", tiers.SEEN), ("A/B-offline", tiers.AB_OFFLINE), ("untouched", tiers.UNTOUCHED))


# ---- reading the runs ----------------------------------------------------------------------------
def read_json(run_dir, name):
    try:
        with open(os.path.join(ROOT, run_dir, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load(out):
    """{arm: {(game, seed): run}} for the runs in the sweep's manifest (the last row of a run wins)."""
    runs = {}
    path = os.path.join(out, "manifest.tsv")
    for row in (_manifest.read_manifest(path) if os.path.exists(path) else []):
        d = row["run_dir"]
        m, end = read_json(d, "metrics.json"), read_json(d, "run_end.json")
        if "levels_completed" not in m or not end:
            continue
        runs.setdefault(row["arm"], {})[row["game"], int(row["seed"])] = dict(
            dir=d, levels=int(m["levels_completed"]), ups=sorted(int(a) for a, _ in (m.get("levelup_events") or [])),
            actions=int(end["n_actions"]), seconds=float(end["elapsed_s"]), end=end["termination"],
            cfg=read_json(d, "run_config.json"), tr=read_json(d, "train_stats.json"),
            kp=read_json(d, "kp/kp_stats.json"))
    return runs


def level_by(run, t):
    """Levels completed by action t (by the cap: the engine's count)."""
    return run["levels"] if t >= T.CAP else sum(1 for a in run["ups"] if a <= t)


def total(runs, arm, pairs, t=T.CAP):
    return sum(level_by(runs[arm][p], t) for p in pairs)


# ---- the rule --------------------------------------------------------------------------------------
def chance_gap(differences, reps=10000, seed=0):
    """The 97.5th percentile of the gap between two totals of the same agent (module doc)."""
    rng, d = random.Random(seed), list(differences)
    sums = sorted(sum(rng.choice(d) * rng.choice((-1, 1)) for _ in d) for _ in range(reps))
    return sums[int(0.975 * reps) - 1]


def rule(runs, pairs):
    """The test's rule on the (game, seed) pairs every arm has."""
    score = {arm: total(runs, arm, pairs) for arm in ("SOT", "SOT-KP", "SOT2")}
    bar = chance_gap([runs["SOT2"][p]["levels"] - runs["SOT"][p]["levels"] for p in pairs])
    gap = score["SOT-KP"] - score["SOT"]
    above_both = score["SOT-KP"] > score["SOT"] and score["SOT-KP"] > score["SOT2"]
    return dict(score=score, gap=gap, bar=bar, above_both=above_both, clears=above_both and gap > bar)


def against(runs, new, base, pairs, t=T.CAP):
    """`new` minus `base`: the total, pairs better / same / worse, games worse on every seed, and the
    95% interval with games resampled (each game keeps its seeds)."""
    d = {p: level_by(runs[new][p], t) - level_by(runs[base][p], t) for p in pairs}
    per_game = {}
    for (game, _), x in d.items():
        per_game.setdefault(game, []).append(x)
    lo, hi = L.bootstrap_delta([sum(v) for v in per_game.values()]) if d else (0, 0)
    return dict(delta=sum(d.values()), better=sum(x > 0 for x in d.values()), same=sum(x == 0 for x in d.values()),
                worse=sum(x < 0 for x in d.values()), lo=lo, hi=hi,
                worse_on_every_seed=sorted(g for g, v in per_game.items() if all(x < 0 for x in v)))


def races(runs, new, base, pairs):
    """Level by level, which arm finished the level in fewer actions (tools/levels_by_budget.py)."""
    side = lambda arm: {p: ([(a, i + 1) for i, a in enumerate(runs[arm][p]["ups"])], runs[arm][p]["actions"]) for p in pairs}
    return L.level_races(side(base), side(new))


# ---- the tables ------------------------------------------------------------------------------------
def md(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(head, rows):
    return [md(head), md(["---"] * len(head))] + [md(r) for r in rows] + [""]


def human_baselines(game):
    """The game's own per-level human action counts, where its metadata carries them."""
    import glob
    for path in glob.glob(os.path.join(ROOT, "environment_files", game, "*", "metadata.json")):
        with open(path) as f:
            return json.load(f).get("baseline_actions") or []
    return []


def planner_rows(kp_runs):
    """Per game: what the planner did in the SOT-KP runs."""
    rows = []
    for game in sorted({g for g, _ in kp_runs}):
        mine = [r for (g, _), r in sorted(kp_runs.items()) if g == game]
        stats = [r["kp"] for r in mine if r["kp"]]
        if not stats or all(s["dormant"] for s in stats):
            rows.append([game, len(mine), "dormant: no click action", "", "", "", "", sum(s["kp_errors"] for s in stats)])
            continue
        why = {}
        for s in stats:
            for v in s["levels"]:
                if v["gave_up"]:
                    why[v["gave_up"]] = why.get(v["gave_up"], 0) + 1
        planned = {}        # level -> actions it took, for the levels cleared by planning
        for r in mine:
            levels = (r["kp"] or {}).get("levels", [])
            for v in levels:        # a won game's last level is left at its last move, not one step after it
                if v["ended"] == "cleared by planning":
                    planned.setdefault(v["level"], []).append(
                        v["left_at"] - v["entered_at"] + (r["end"] == "win" and v is levels[-1]))
        human = human_baselines(game)
        per_level = "; ".join(f"L{k}: {st.median(v):.0f} ({len(v)}/{len(mine)} runs"
                              + (f", human {human[k - 1]}" if k <= len(human) else "") + ")" for k, v in sorted(planned.items()))
        rows.append([game, len(mine), sum(s["levels_by_planning"] for s in stats), sum(s["levels_by_goose"] for s in stats),
                     " / ".join(str(sum(s["clicks"][k] for s in stats)) for k in ("probe", "verify", "plan")),
                     "; ".join(f"{k} ({n})" for k, n in sorted(why.items(), key=lambda kv: -kv[1])) or "-",
                     per_level or "-", sum(s["kp_errors"] for s in stats)])
    return rows


def funnel_rows(kp_runs):
    games = {}
    for (game, _), r in sorted(kp_runs.items()):
        try:
            with open(os.path.join(ROOT, r["dir"], "kp", "events.jsonl")) as f:
                events = [json.loads(line) for line in f]
        except OSError:
            continue
        if r["kp"] and not r["kp"]["dormant"]:
            games.setdefault(game, []).extend(kp_offline.level_steps(r["kp"], events))
    return [[g] + [sum(v["reached"][k] for v in levels) for k in range(len(kp_offline.STEPS))]
            + [sum(v["ended"] == "cleared by Goose" for v in levels), sum(v["ended"] == "not cleared" for v in levels)]
            for g, levels in sorted(games.items()) if levels]


def sot_flags(cfg):
    return (cfg.get("label") == "novel" and bool(cfg.get("return_map")) and cfg.get("train_every") == 1
            and cfg.get("train_recent") == 16 and cfg.get("upgrades") == "attempt,bars,map_sched,objclick")


def write(out, partial=False):
    """The verdict page of one sweep. Returns its text."""
    runs, jobs = load(out), T.sweep_jobs()
    arms = [a for a in T.ARM_NAMES if a in runs]
    wanted = sorted({(j.game, j.seed) for j in jobs})
    pairs = [p for p in wanted if all(p in runs.get(a, {}) for a in T.ARM_NAMES)]
    missing = len(jobs) - sum(len(runs.get(a, {})) for a in T.ARM_NAMES)
    is_test = not T.DEV
    lines = [f"# {T.NAME[0].upper() + T.NAME[1:]}: {'the verdict' if is_test else 'the tables (exploratory: it has no rule)'}", "",
             f"Sweep `{os.path.relpath(out, ROOT)}`. {len(pairs)} of {len(wanted)} game-and-seed pairs have every arm; "
             f"{missing} of {len(jobs)} runs are missing.", ""]
    if not pairs or (missing and not partial):
        lines.append(f"**No verdict and no table: {missing} runs are not in the manifest yet.**"
                     + ("" if not pairs else " (`--partial` shows the tables on the pairs that are complete.)"))
        return finish(out, lines, runs)

    # 1. the rule (the test only, and only when every run is in)
    if is_test:
        r = rule(runs, pairs)
        verdict = ("CLEARS THE NOISE" if r["clears"] else "INSIDE THE NOISE at this budget") if not missing else \
            "NOT GIVEN: the sweep is not complete"
        lines += ["## The rule", ""] + table(["Arm", "Levels by 5,000 actions"], [[a, r["score"][a]] for a in T.ARM_NAMES])
        lines += [f"- SOT-KP above both baselines: **{'yes' if r['above_both'] else 'no'}** "
                  f"({r['score']['SOT-KP']} against {r['score']['SOT']} and {r['score']['SOT2']}).",
                  f"- SOT-KP - SOT = **{r['gap']:+d}**; the chance gap's 97.5th percentile is **{r['bar']}** levels "
                  f"(from the {len(pairs)} differences SOT2 - SOT, which sum to "
                  f"{r['score']['SOT2'] - r['score']['SOT']:+d}): {'above' if r['gap'] > r['bar'] else 'not above'} it.",
                  "", f"**The planner: {verdict}.**", ""]
        for arm in ("SOT", "SOT2"):
            if not ANCHOR[0] <= r["score"][arm] * len(wanted) / len(pairs) <= ANCHOR[1]:
                lines += [f"**Sanity anchor: {arm} ended on {r['score'][arm]} levels, outside {ANCHOR[0]} to {ANCHOR[1]}. "
                          f"Check the set-up (engine mode, flags) before reading anything here.**", ""]
        lines += [f"Upgrade screen 3, another seed, for scale: {SCREEN3}.", ""]

    # 2. reported, not part of the rule
    lines += ["## Reported, not part of the rule", "", "### Levels by budget, and by kind of game", ""]
    clicks = [p for p in pairs if p[0] not in kp_offline.NO_CLICKS]
    groups = [("all games", pairs), ("click games", clicks), ("games with no click (the arms are the same agent)",
                                                               [p for p in pairs if p not in clicks])]
    groups += [(f"tier: {name}", [p for p in pairs if p[0] in games]) for name, games in TIER_GROUPS]
    lines += table(["Games", "Pairs"] + [f"{a} by {t:,}" for t in BUDGETS for a in arms],
                   [[name, len(ps)] + [total(runs, a, ps, t) for t in BUDGETS for a in arms] for name, ps in groups if ps])
    lines += ["### Pair by pair", ""]
    versus = [("SOT-KP", "SOT")] + ([("SOT-KP", "SOT2"), ("SOT2", "SOT")] if is_test else [])
    rows = []
    for new, base in versus:
        v = against(runs, new, base, pairs)
        rows.append([f"{new} - {base}", f"{v['delta']:+d}", f"[{v['lo']:+.0f}, {v['hi']:+.0f}]",
                     f"{v['better']} / {v['same']} / {v['worse']}", ", ".join(v["worse_on_every_seed"]) or "none"])
    lines += table(["Difference", "Levels", "95% interval, games resampled", "Pairs better / same / worse",
                    "Games worse on every seed"], rows)
    lines += ["### Every game and seed", ""]
    lines += table(["Game", "Tier", "Seed"] + arms + ["Level-ups at action: " + " ; ".join(arms)],
                   [[g, tiers.tier_of(g), s] + [runs[a][g, s]["levels"] for a in arms]
                    + [" ; ".join(" ".join(map(str, runs[a][g, s]["ups"])) or "-" for a in arms)] for g, s in pairs])
    lines += ["### Level races: who finished a level in fewer actions", "",
              "| Levels | Races | Games | New faster | Base faster | Same or undecided | Net | 95% interval | Median ratio |",
              "|---|---|---|---|---|---|---|---|---|"]
    for new, base in versus:
        rs = races(runs, new, base, clicks)
        lines += [L.race_row(f"{new} against {base}, click games, level 1", [r for r in rs if r[2] == 1]),
                  L.race_row(f"{new} against {base}, click games, levels 2 and up", [r for r in rs if r[2] >= 2])]
    lines.append("")

    # 3. what the planner did
    kp = {p: runs["SOT-KP"][p] for p in pairs}
    lines += ["### What the planner did (SOT-KP runs)", ""]
    lines += table(["Game", "Runs", "Levels cleared by planning", "... by Goose", "Clicks: probe / verify / plan",
                    "Why it left a level to Goose", "Actions for a planned level: median (runs, human baseline)",
                    "Errors"], planner_rows(kp))
    lines += ["### The funnel, over the levels after the first", ""]
    lines += table(["Game"] + [s.capitalize() for s in kp_offline.STEPS] + ["Cleared by Goose after a fallback", "Not cleared"],
                   funnel_rows(kp))

    # 4. are the runs what they should be
    every = [r for a in arms for r in runs[a].values()]
    ms = {a: st.median(1000 * r["seconds"] / r["actions"] for r in runs[a].values()) for a in arms}
    wrong = sorted(r["dir"] for r in every if not sot_flags(r["cfg"]))
    kp_wrong = sorted(r["dir"] for a in arms for r in runs[a].values()
                      if (r["cfg"].get("kp_config_hash") == T.kp_config_hash()) != (a == "SOT-KP"))
    lines += ["### Are the runs what they should be", "",
              f"- Ended at the cap or at a win: {sum(r['end'] in ('cap', 'win') for r in every)} of {len(every)} runs.",
              f"- SOT's flags in run_config.json: {len(every) - len(wrong)} of {len(every)}"
              + (f" (not: {', '.join(wrong[:5])})." if wrong else "."),
              f"- The planner on in the SOT-KP runs only, with the committed settings: "
              f"{'yes' if not kp_wrong else 'NO: ' + ', '.join(kp_wrong[:5])}.",
              f"- A training step per action, lowest run: "
              f"{min(r['tr'].get('train_steps', 0) / r['actions'] for r in every):.0%}.",
              f"- Errors inside the planner: {sum((r['kp'] or {}).get('kp_errors', 0) for r in kp.values())} "
              f"in {len(kp)} runs.",
              "- Median time per action: " + ", ".join(f"{a} {ms[a]:.0f} ms" for a in arms)
              + f"; a run took {st.median(r['seconds'] for r in every) / 60:.1f} minutes at the median.", ""]
    return finish(out, lines, runs)


def finish(out, lines, runs):
    text = "\n".join(lines)
    with open(os.path.join(out, "verdict.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "game", "tier", "seed", "levels", "levelup_actions", "levels_by_1000", "levels_by_2000",
                    "actions", "seconds", "ended", "levels_by_planning", "levels_by_goose", "kp_errors", "run_folder"])
        for arm in sorted(runs):
            for (game, seed), r in sorted(runs[arm].items()):
                kp = r["kp"] or {}
                w.writerow([arm, game, tiers.tier_of(game), seed, r["levels"], ";".join(map(str, r["ups"])),
                            level_by(r, 1000), level_by(r, 2000), r["actions"], round(r["seconds"]), r["end"],
                            kp.get("levels_by_planning", ""), kp.get("levels_by_goose", ""), kp.get("kp_errors", ""), r["dir"]])
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", default="", help="a sweep folder (default: the current one)")
    ap.add_argument("--partial", action="store_true", help="tables for a sweep that is not complete (never a verdict)")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.sweep) if a.sweep else T.current()
    print(write(out, a.partial))
    print(f"\nwrote {os.path.relpath(out, ROOT)}/verdict.md and runs.csv")


if __name__ == "__main__":
    main()
