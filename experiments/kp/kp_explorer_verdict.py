#!/usr/bin/env python3
"""kp_explorer_verdict.py - the two rules of the explorer's test, and the tables read beside them.

The problem it fixes: the test's rules were written before its runs
(docs/4-knowledge-planner/kp-explorer-test.md). This applies them the same way
every time, from the sweep's manifest, and gives no verdict before every run
is in. It is the planner's rule file (kp_verdict.py) again for other arms,
with one rule more; that file is left as the planner's test used it.

    E21TEST=1 uv run python experiments/kp/kp_test.py verdict             the verdict and the tables
    E21TEST=1 uv run python experiments/kp/kp_explorer_verdict.py [--sweep results/kp/test_e21/<stamp>] [--partial]

It writes <sweep>/verdict.md and <sweep>/runs.csv and prints the page.

The arms: SOT-KP (Goose with upgrade screen 3's upgrades and the planner), SOT-KP21 (the same with the
explorer E21 on the levels after the first of games with keys), SOT-KP2 (SOT-KP again).

RULE A, more levels (the planner's test's rule):
  score             an arm's levels completed by 5,000 actions, summed over its runs
  clears the noise  SOT-KP21's score is above both SOT-KP's and SOT-KP2's, AND SOT-KP21 - SOT-KP is
                    above the 97.5th percentile of the chance gap
  the chance gap    how far two totals of the SAME agent come apart: this test's own differences
                    SOT-KP2 - SOT-KP, one per game and seed, drawn again with replacement, each with a
                    sign at random, and summed; 10,000 times, seed 0

RULE B, sooner, where the explorer acts (the levels after the first of the 18 games with keys):
  a level race      a level both runs of a pair started and at least one finished: the run that took
                    fewer actions on it wins it (tools/levels_by_budget.py, level_races)
  net               races the first arm won minus races the second won, levels 2 and up, games with keys
  clears the noise  the net of SOT-KP21 against SOT-KP and against SOT-KP2 are both above zero, AND
                    the net against SOT-KP is above the 97.5th percentile of the chance gap built, as
                    in rule A, from the nets of SOT-KP2 against SOT-KP, one per game and seed

Everything else is reported and is not part of a rule.
"""
import argparse
import csv
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "custom_agents"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kp_bridge  # noqa: E402  (the settings' hashes)
import kp_test as T  # noqa: E402
import kp_verdict as V  # noqa: E402  (the planner's rule file: the runs, the chance gap, the paired tables)
import levels_by_budget as L  # noqa: E402
import tiers  # noqa: E402

BASE, NEW, AGAIN = "SOT-KP", "SOT-KP21", "SOT-KP2"
# The games on which Goose can send a key (it pressed one in the planner's test), where the explorer may act.
# On the other seven Goose sends clicks only and SOT-KP21 is SOT-KP: E21 calls the planner itself.
KEYS = ("ar25", "bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20", "m0r0", "re86", "sb26", "sc25", "sk48",
        "sp80", "tr87", "tu93", "wa30")
GROUPMATE = ("ar25", "bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52")    # the games E21's author ran it on
ANCHOR = (40, 80)           # SOT-KP or SOT-KP2 outside this many levels on 50 runs: check the set-up
SEEN_BEFORE = "SOT-KP 62 levels on seeds 11 and 12 (the planner's test)"


def rule_levels(runs, pairs):
    """Rule A on the (game, seed) pairs every arm has."""
    score = {arm: V.total(runs, arm, pairs) for arm in (BASE, NEW, AGAIN)}
    bar = V.chance_gap([runs[AGAIN][p]["levels"] - runs[BASE][p]["levels"] for p in pairs])
    gap = score[NEW] - score[BASE]
    above_both = score[NEW] > score[BASE] and score[NEW] > score[AGAIN]
    return dict(score=score, gap=gap, bar=bar, above_both=above_both, clears=above_both and gap > bar)


def sooner(runs, new, base, pairs):
    """{(game, seed): net} over the levels after the first: +1 for each level `new` finished in fewer
    actions than `base`, -1 for each `base` finished in fewer."""
    net = {p: 0 for p in pairs}
    for game, seed, level, winner, _ in V.races(runs, new, base, pairs):
        if level >= 2:
            net[game, seed] += (winner == "new") - (winner == "base")
    return net


def rule_sooner(runs, pairs):
    """Rule B on the pairs of the games with keys."""
    keys = [p for p in pairs if p[0] in KEYS]
    net = sum(sooner(runs, NEW, BASE, keys).values())
    net_again = sum(sooner(runs, NEW, AGAIN, keys).values())
    control = sooner(runs, AGAIN, BASE, keys)
    bar = V.chance_gap(list(control.values())) if keys else 0
    return dict(net=net, net_again=net_again, control=sum(control.values()), bar=bar, pairs=len(keys),
                clears=net > 0 and net_again > 0 and net > bar)


def explorer_rows(new_runs):
    """Per game with keys: what the explorer did in the SOT-KP21 runs."""
    rows = []
    for game in sorted({g for g, _ in new_runs if g in KEYS}):
        stats = [r["kp"] for (g, _), r in sorted(new_runs.items()) if g == game and r["kp"]]
        later = [v for s in stats for v in s.get("levels", []) if v["level"] >= 2]
        moves = lambda who: sum(v.get("moves", {}).get(who, 0) for v in later)
        rows.append([game, tiers.tier_of(game), "yes" if game in GROUPMATE else "no", len(stats),
                     sum(any(v["level"] >= 2 for v in s.get("levels", [])) for s in stats), len(later),
                     sum(v["ended"] == "cleared by the explorer" for v in later),
                     sum(v["ended"] == "cleared by Goose" for v in later), moves("explorer"), moves("goose"),
                     sum(s.get("kp_errors", 0) for s in stats)])
    return rows


def write(out, partial=False):
    """The verdict page of one sweep. Returns its text."""
    runs, jobs = V.load(out), T.sweep_jobs()
    arms = [a for a in T.ARM_NAMES if a in runs]
    wanted = sorted({(j.game, j.seed) for j in jobs})
    pairs = [p for p in wanted if all(p in runs.get(a, {}) for a in T.ARM_NAMES)]
    missing = len(jobs) - sum(len(runs.get(a, {})) for a in T.ARM_NAMES)
    lines = [f"# {T.NAME[0].upper() + T.NAME[1:]}: the verdict", "",
             f"Sweep `{os.path.relpath(out, V.ROOT)}`. {len(pairs)} of {len(wanted)} game-and-seed pairs have every arm; "
             f"{missing} of {len(jobs)} runs are missing.", ""]
    if not pairs or (missing and not partial):
        lines.append(f"**No verdict and no table: {missing} runs are not in the manifest yet.**"
                     + ("" if not pairs else " (`--partial` shows the tables on the pairs that are complete.)"))
        return finish(out, lines, runs)

    # 1. the two rules (a verdict only when every run is in)
    a, b = rule_levels(runs, pairs), rule_sooner(runs, pairs)
    say = lambda r: ("CLEARS THE NOISE" if r["clears"] else "INSIDE THE NOISE at this budget") if not missing else \
        "NOT GIVEN: the sweep is not complete"
    lines += ["## Rule A: more levels", ""] + V.table(["Arm", "Levels by 5,000 actions"], [[x, a["score"][x]] for x in T.ARM_NAMES])
    lines += [f"- {NEW} above both {BASE} and {AGAIN}: **{'yes' if a['above_both'] else 'no'}** "
              f"({a['score'][NEW]} against {a['score'][BASE]} and {a['score'][AGAIN]}).",
              f"- {NEW} - {BASE} = **{a['gap']:+d}**; the chance gap's 97.5th percentile is **{a['bar']}** levels "
              f"(from the {len(pairs)} differences {AGAIN} - {BASE}, which sum to "
              f"{a['score'][AGAIN] - a['score'][BASE]:+d}): {'above' if a['gap'] > a['bar'] else 'not above'} it.",
              "", f"**The explorer, more levels: {say(a)}.**", "",
              "## Rule B: sooner, on the levels after the first of the games with keys", "",
              f"- Level races won minus lost, {NEW} against {BASE}: **{b['net']:+d}**; against {AGAIN}: "
              f"**{b['net_again']:+d}** (both must be above zero).",
              f"- The same for {AGAIN} against {BASE}, two runs of one agent: {b['control']:+d}. The chance gap's 97.5th "
              f"percentile, from its {b['pairs']} nets: **{b['bar']}**: {b['net']:+d} is "
              f"{'above' if b['net'] > b['bar'] else 'not above'} it.",
              "", f"**The explorer, sooner: {say(b)}.**", ""]
    for arm in (BASE, AGAIN):
        if not ANCHOR[0] <= a["score"][arm] * len(wanted) / len(pairs) <= ANCHOR[1]:
            lines += [f"**Sanity anchor: {arm} ended on {a['score'][arm]} levels, outside {ANCHOR[0]} to {ANCHOR[1]}. "
                      f"Check the set-up (engine mode, flags) before reading anything here.**", ""]
    lines += [f"On other seeds, for scale: {SEEN_BEFORE}.", ""]

    # 2. reported, not part of a rule
    keys = [p for p in pairs if p[0] in KEYS]
    groups = [("all games", pairs), ("games with keys (the explorer may act)", keys),
              ("games with clicks only (the arms are the same agent)", [p for p in pairs if p not in keys]),
              ("games with keys: this project's dev and seen", [p for p in keys if p[0] in tiers.BUILD_FROM]),
              ("games with keys: held-out here, used by E21's author", [p for p in keys if p[0] in GROUPMATE
                                                                         and p[0] not in tiers.BUILD_FROM]),
              ("games with keys: held-out here, used by nobody for E21", [p for p in keys if p[0] not in GROUPMATE
                                                                           and p[0] not in tiers.BUILD_FROM])]
    groups += [(f"tier: {name}", [p for p in pairs if p[0] in games]) for name, games in V.TIER_GROUPS]
    lines += ["## Reported, not part of a rule", "", "### Levels by budget, and by kind of game", ""]
    lines += V.table(["Games", "Pairs"] + [f"{x} by {t:,}" for t in V.BUDGETS for x in arms],
                     [[name, len(ps)] + [V.total(runs, x, ps, t) for t in V.BUDGETS for x in arms] for name, ps in groups if ps])
    versus = [(NEW, BASE), (NEW, AGAIN), (AGAIN, BASE)]
    rows = []
    for name, ps in (("all games", pairs), ("games with keys", keys)):
        for new, base in versus:
            v = V.against(runs, new, base, ps)
            rows.append([name, f"{new} - {base}", f"{v['delta']:+d}", f"[{v['lo']:+.0f}, {v['hi']:+.0f}]",
                         f"{v['better']} / {v['same']} / {v['worse']}", ", ".join(v["worse_on_every_seed"]) or "none"])
    lines += ["### Pair by pair", ""]
    lines += V.table(["Games", "Difference", "Levels", "95% interval, games resampled", "Pairs better / same / worse",
                      "Games worse on every seed"], rows)
    lines += ["### Every game and seed", ""]
    lines += V.table(["Game", "Tier", "Keys", "Seed"] + arms + ["Level-ups at action: " + " ; ".join(arms)],
                     [[g, tiers.tier_of(g), "yes" if g in KEYS else "no", s] + [runs[x][g, s]["levels"] for x in arms]
                      + [" ; ".join(" ".join(map(str, runs[x][g, s]["ups"])) or "-" for x in arms)] for g, s in pairs])
    lines += ["### Level races: who finished a level in fewer actions", "",
              "| Levels | Races | Games | New faster | Base faster | Same or undecided | Net | 95% interval | Median ratio |",
              "|---|---|---|---|---|---|---|---|---|"]
    for new, base in versus:
        every, with_keys = V.races(runs, new, base, pairs), V.races(runs, new, base, keys)
        lines += [L.race_row(f"{new} against {base}, all games, level 1", [r for r in every if r[2] == 1]),
                  L.race_row(f"{new} against {base}, games with keys, levels 2 and up", [r for r in with_keys if r[2] >= 2])]
    lines.append("")

    # 3. what the explorer and the planner did
    lines += [f"### What the explorer did ({NEW} runs, games with keys)", ""]
    lines += V.table(["Game", "Tier", "Used by E21's author", "Runs", "Runs with a level 1", "Levels entered after the first",
                      "Cleared by the explorer", "Cleared by Goose", "The explorer's moves on them", "Goose's moves on them",
                      "Errors"], explorer_rows({p: runs[NEW][p] for p in pairs}))
    by = lambda arm, k: sum((runs[arm][p]["kp"] or {}).get(k, 0) for p in pairs)
    lines += ["### Who cleared the levels", ""]
    lines += V.table(["Arm", "By planning (E20)", "By the explorer (E21)", "All levels"],
                     [[x, by(x, "levels_by_planning"), by(x, "levels_by_explorer"), V.total(runs, x, pairs)] for x in arms])

    # 4. are the runs what they should be
    every = [r for x in arms for r in runs[x].values()]
    wrong = sorted(r["dir"] for r in every if not V.sot_flags(r["cfg"]))
    hashes = {BASE: T.kp_config_hash(), AGAIN: T.kp_config_hash(),
              NEW: kp_bridge.digest(kp_bridge.load_settings(os.path.join(ROOT, T.KP_E21_CONFIG)))}
    cfg_wrong = sorted(r["dir"] for x in arms for r in runs[x].values() if r["cfg"].get("kp_config_hash") != hashes[x])
    keys_wrong = sorted(r["dir"] for (g, _), r in runs[NEW].items()
                        if r["kp"] and r["kp"].get("keys") != (g in KEYS))
    ms = {x: st.median(1000 * r["seconds"] / r["actions"] for r in runs[x].values()) for x in arms}
    lines += ["### Are the runs what they should be", "",
              f"- Ended at the cap or at a win: {sum(r['end'] in ('cap', 'win') for r in every)} of {len(every)} runs.",
              f"- SOT's flags in run_config.json: {len(every) - len(wrong)} of {len(every)}"
              + (f" (not: {', '.join(wrong[:5])})." if wrong else "."),
              f"- Each arm with its own settings (the planner's for {BASE} and {AGAIN}, the explorer's for {NEW}): "
              f"{'yes' if not cfg_wrong else 'NO: ' + ', '.join(cfg_wrong[:5])}.",
              f"- In the {NEW} runs the explorer's path was taken on the 18 games with keys and on no other: "
              f"{'yes' if not keys_wrong else 'NO: ' + ', '.join(keys_wrong[:5])}.",
              f"- Errors inside the planner's files: {sum((r['kp'] or {}).get('kp_errors', 0) for r in every)} "
              f"in {len(every)} runs.",
              "- Median time per action: " + ", ".join(f"{x} {ms[x]:.0f} ms" for x in arms)
              + f"; a run took {st.median(r['seconds'] for r in every) / 60:.1f} minutes at the median.", ""]
    return finish(out, lines, runs)


def finish(out, lines, runs):
    text = "\n".join(lines)
    with open(os.path.join(out, "verdict.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "game", "tier", "keys", "seed", "levels", "levelup_actions", "levels_by_1000", "levels_by_2000",
                    "actions", "seconds", "ended", "levels_by_planning", "levels_by_explorer", "kp_errors", "run_folder"])
        for arm in sorted(runs):
            for (game, seed), r in sorted(runs[arm].items()):
                kp = r["kp"] or {}
                w.writerow([arm, game, tiers.tier_of(game), int(game in KEYS), seed, r["levels"], ";".join(map(str, r["ups"])),
                            V.level_by(r, 1000), V.level_by(r, 2000), r["actions"], round(r["seconds"]), r["end"],
                            kp.get("levels_by_planning", ""), kp.get("levels_by_explorer", ""), kp.get("kp_errors", ""),
                            r["dir"]])
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", default="", help="a sweep folder (default: the current one)")
    ap.add_argument("--partial", action="store_true", help="tables for a sweep that is not complete (never a verdict)")
    a = ap.parse_args()
    if not T.E21TEST:
        sys.exit("this is the explorer's test's rule: put E21TEST=1 in front")
    out = os.path.join(ROOT, a.sweep) if a.sweep else T.current()
    print(write(out, a.partial))
    print(f"\nwrote {os.path.relpath(out, ROOT)}/verdict.md and runs.csv")


if __name__ == "__main__":
    main()
