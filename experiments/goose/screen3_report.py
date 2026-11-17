#!/usr/bin/env python3
"""screen3_report.py - the tables of upgrade screen 3, read from its manifest.

The problem it fixes: 225 runs are nine columns of numbers, and the rule they are
read by was written before the runs (docs/1-goose-upgrades/upgrade-screen3.md,
"The rule"). This applies that rule the same way every time, also to a sweep that
is only partly done, and keeps one row per run as a CSV because results/ is not
in git.

    uv run python experiments/goose/screen3.py report [--sweep results/screen3/<stamp>] [--docs]

writes <sweep>/report.md and <sweep>/runs.csv (and, with --docs, the CSV again as
docs/1-goose-upgrades/upgrade-screen3-runs.csv), and prints the short version.

THE RULE, as the plan page has it:
  score            an arm's levels completed by 5,000 actions, summed over the games
                   (the engine's count, as compute_metrics.py reports it)
  rank             by score; ties go to more levels by 2,000 actions, then to more
                   games better than `base`
  clears the noise score above both `base` and `base2`, AND the 95% bootstrap interval
                   of its difference from `base` (games resampled, as
                   tools/paired_compare.py does) is above 0. Otherwise: inside the noise
  main effect      mean score of the four arms that have an upgrade minus the mean of
                   the four that do not (`base2` left out), with the same interval
  best combo       the top-ranked of the seven upgraded arms
  sanity anchor    `base` should land near 15 levels; under 8 or over 24: check the setup

Arms are compared on the games every arm has finished, so a partial sweep is still
paired by game; the report says which games those are. It gives no adoption
verdict: the screen is exploratory (one seed, 5,000 actions).
"""
import csv
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "tools"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import levels_by_budget as L  # noqa: E402  (level races, AULC; and paired_compare's bootstrap)
import manifest as _manifest  # noqa: E402
import tiers  # noqa: E402
import screen3 as S  # noqa: E402

BUDGETS = (1000, 2000, 5000)
ANCHOR = (8, 24)            # `base` outside this many levels on 25 games: check the setup
TIER_GROUPS = (("dev and seen", tiers.DEV + tiers.SEEN), ("A/B-offline", tiers.AB_OFFLINE),
               ("untouched", tiers.UNTOUCHED))
SCHEDULES = ("stay200", "walk200", "stay30", "walk30", "sweep30")
DOCS_CSV = "docs/1-goose-upgrades/upgrade-screen3-runs.csv"


# ---- reading the runs ----------------------------------------------------------------------------
def read_json(run_dir, name):
    path = os.path.join(ROOT, run_dir, name)
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def load(out):
    """{arm: {game: run}} for the runs in the sweep's manifest (the last row of a run wins)."""
    runs = {}
    path = os.path.join(out, "manifest.tsv")
    for row in (_manifest.read_manifest(path) if os.path.exists(path) else []):
        d = row["run_dir"]
        m, end = read_json(d, "metrics.json"), read_json(d, "run_end.json")
        if "levels_completed" not in m:
            continue
        try:
            commit = open(os.path.join(ROOT, os.path.dirname(d), "git_info.txt")).readline().split(":", 1)[1].strip()
        except (OSError, IndexError):
            commit = "?"
        runs.setdefault(row["arm"], {})[row["game"]] = dict(
            dir=d, levels=int(m["levels_completed"]), events=m.get("levelup_events") or [],
            ups=sorted(int(a) for a, _ in (m.get("levelup_events") or [])),
            taken=int(m.get("actions_taken") or m["n_actions"]), n=int(m["n_actions"]),
            unique=int(m["unique_states"]), aps=end.get("actions_per_sec") or m.get("actions_per_sec") or 0.0,
            ms_med=m.get("wall_ms_med"), end=m.get("termination"), cfg=read_json(d, "run_config.json"),
            up=read_json(d, "upgrades_stats.json"), mp=read_json(d, "return_map_stats.json"),
            tr=read_json(d, "train_stats.json"), commit=commit)
    return runs


def level_by(run, t):
    """Levels completed by action t (by the cap: the engine's count)."""
    return run["levels"] if t >= S.CAP else sum(1 for a in run["ups"] if a <= t)


# ---- the rule --------------------------------------------------------------------------------------
def total(runs, arm, games, t=S.CAP):
    return sum(level_by(runs[arm][g], t) for g in games)


def against_base(runs, arm, games, t=S.CAP, base="base"):
    """The arm's levels minus base's, with better/same/worse games and the 95% interval."""
    d = [level_by(runs[arm][g], t) - level_by(runs[base][g], t) for g in games]
    lo, hi = L.bootstrap_delta(d) if d else (0, 0)
    return dict(delta=sum(d), lo=lo, hi=hi, better=sum(x > 0 for x in d), same=sum(x == 0 for x in d),
                worse=sum(x < 0 for x in d))


def clears_noise(score, base, base2, lo):
    """Above both runs of the baseline, and the interval of the difference from base above 0."""
    return base2 is not None and score > base and score > base2 and lo > 0


def rank(runs, games):
    """The upgraded arms, best first: score, then levels by 2,000 actions, then games better than base."""
    arms = [a for a in S.UPGRADED if a in runs]
    key = lambda a: (-total(runs, a, games), -total(runs, a, games, 2000), -against_base(runs, a, games)["better"])
    return sorted(arms, key=key)


def main_effect(runs, upgrade, games, t=S.CAP):
    """Mean score of the four arms with `upgrade` minus the mean of the four without
    (base2 left out), and its 95% interval; None unless all eight arms are there."""
    eight = [a for a in S.TABLE_ORDER if a != "base2"]
    if any(a not in runs for a in eight):
        return None
    yes = [a for a in eight if upgrade in S.ARMS[a][2]]
    no = [a for a in eight if upgrade not in S.ARMS[a][2]]
    d = [st.mean(level_by(runs[a][g], t) for a in yes) - st.mean(level_by(runs[a][g], t) for a in no) for g in games]
    lo, hi = L.bootstrap_delta(d) if d else (0.0, 0.0)
    return dict(effect=sum(d), lo=lo, hi=hi, yes=yes, no=no)


def added_to(runs, upgrade, games):
    """What `upgrade` adds to each of the four arms without it: [(arm, arm + upgrade, levels added)]."""
    eight = [a for a in S.TABLE_ORDER if a != "base2"]
    out = []
    for a in eight:
        if upgrade not in S.ARMS[a][2]:
            b = next(x for x in eight if set(S.ARMS[x][2]) == set(S.ARMS[a][2]) | {upgrade})
            out.append((a, b, total(runs, b, games) - total(runs, a, games)))
    return out


def against_both(runs, arm, games):
    """The arm's levels minus the mean of the two baselines', with the 95% interval."""
    d = [runs[arm][g]["levels"] - (runs["base"][g]["levels"] + runs["base2"][g]["levels"]) / 2 for g in games]
    lo, hi = L.bootstrap_delta(d) if d else (0.0, 0.0)
    return sum(d), lo, hi


def effects_agree(combo, effects):
    """Do the main effects point at the same combination: positive for each upgrade the
    combo has, not positive for each it lacks?"""
    return all((effects[u]["effect"] > 0) == (u in combo) for u in "SOT")


# ---- the tables ------------------------------------------------------------------------------------
def table(head, rows, right=()):
    """A markdown table; `right`: the columns (by number) that hold numbers."""
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---:" if i in right else "---" for i in range(len(head))) + "|"]
    return lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def as_races(runs, arm, games):
    """One arm's runs in the form tools/levels_by_budget.py compares."""
    return {(g, str(S.SEED)): (runs[arm][g]["events"], runs[arm][g]["taken"]) for g in games}


def race_counts(rows):
    new, base = sum(r[3] == "new" for r in rows), sum(r[3] == "base" for r in rows)
    per_game = {}
    for g, _, _, winner, _ in rows:
        per_game[g] = per_game.get(g, 0) + (winner == "new") - (winner == "base")
    lo, hi = L.bootstrap_delta(list(per_game.values())) if per_game else (0, 0)
    return new, base, len(rows) - new - base, lo, hi


def time_to_level(runs, games, arms):
    """Against base, for each arm: levels by each budget, AULC, and the level races."""
    base = as_races(runs, "base", games)
    shortest = min(min(runs[a][g]["taken"] for g in games) for a in arms + ["base"])
    rows = []
    for arm in arms:
        new = as_races(runs, arm, games)
        by = [L.paired(base, new, lambda run, t=t: L.level_at(run[0], t)) for t in BUDGETS]
        au = L.paired(base, new, lambda run: L.aulc(run[0], L.T_MIN_DEFAULT, float(shortest)))
        races = L.level_races(base, new)
        n1, b1, u1, _, _ = race_counts([r for r in races if r[2] == 1])
        n2, b2, u2, lo, hi = race_counts([r for r in races if r[2] >= 2])
        rows.append([arm] + [f"{r['new'] - r['base']:+d}" for r in by]
                    + [f"{au['new'] - au['base']:+.2f} [{au['lo']:+.2f}, {au['hi']:+.2f}]",
                       f"{n1} / {b1} / {u1}", f"{n2} / {b2} / {u2}", f"{n2 - b2:+d} [{lo:+.0f}, {hi:+.0f}]"])
    head = ["Arm"] + [f"levels by {t:,}, arm - base" for t in BUDGETS] + [
        f"AULC, {L.T_MIN_DEFAULT:g} to {shortest:,} actions, arm - base [95% interval]",
        "level 1: arm faster / base faster / same or undecided",
        "levels 2 and up: arm faster / base faster / same or undecided", "levels 2 and up, arm - base [95% interval]"]
    return table(head, rows, right=(1, 2, 3))


def upgrade_tables(runs, arms):
    """What each upgrade did, summed over each arm's finished runs."""
    out = []
    s_arms = [a for a in arms if "S" in S.ARMS[a][2]]
    if s_arms:
        rows = []
        for a in s_arms:
            mp = [r["mp"] for r in runs[a].values()]
            pulls = [sum(m.get("sched_pulls_" + n, 0) for m in mp) for n in SCHEDULES]
            dec = [sum(m.get("sched_decisions_" + n, 0) for m in mp) for n in SCHEDULES]
            new = [sum(m.get("sched_new_" + n, 0) for m in mp) for n in SCHEDULES]
            moves = sum(r["n"] for r in runs[a].values())
            rows.append([a, " / ".join(map(str, pulls)),
                         " / ".join(f"{d / max(sum(dec), 1):.0%}" for d in dec),
                         " / ".join(f"{n / d:.3f}" if d else "-" for n, d in zip(new, dec)),
                         sum(m.get("untried_first", 0) for m in mp),
                         f"{sum(m.get('map_actions', 0) for m in mp) / max(moves, 1):.1%}"])
        out += ["**S (`map_sched`)**: the five schedules in the order " + ", ".join(SCHEDULES) + ".", ""]
        out += table(["Arm", "pulls", "share of decisions", "new screens per decision", "untried-first presses",
                      "moves chosen by the map"], rows, right=(4, 5)) + [""]
        g_arms = [a for a in arms if "S" not in S.ARMS[a][2]]
        rows = [[a, sum(r["mp"].get("gate_walk", 0) for r in runs[a].values()),
                 sum(r["mp"].get("gate_stay", 0) for r in runs[a].values()),
                 f"{sum(r['mp'].get('map_actions', 0) for r in runs[a].values()) / max(sum(r['n'] for r in runs[a].values()), 1):.1%}"]
                for a in g_arms]
        out += ["For comparison, the arms with `map_gated`:", ""]
        out += table(["Arm", "walk-back choices", "stay choices", "moves chosen by the map"], rows, right=(1, 2, 3)) + [""]
    o_arms = [a for a in arms if "O" in S.ARMS[a][2]]
    if o_arms:
        rows = []
        for a in o_arms:
            up = [r["up"] for r in runs[a].values()]
            red, kept = sum(u.get("objclick_redrawn", 0) for u in up), sum(u.get("objclick_kept", 0) for u in up)
            shapes = sum(u.get("objclick_shapes", 0) for u in up)
            moves = sum(r["n"] for r in runs[a].values())
            rows.append([a, f"{red:,}", f"{kept:,}", f"{(red + kept) / max(moves, 1):.0%}", f"{red / max(moves, 1):.0%}",
                         f"{shapes / red:.1f}" if red else "-",
                         sum(1 for u in up if not u.get("objclick_redrawn") and not u.get("objclick_kept"))])
        out += ["**O (`objclick`)**", ""]
        out += table(["Arm", "clicks redrawn by shape", "clicks kept (Goose's cell)", "moves that were Goose clicks",
                      "moves redrawn", "mean shapes on a redrawn screen", "games with no click"], rows,
                     right=(1, 2, 3, 4, 5, 6)) + [""]
    t_arms = [a for a in arms if "T" in S.ARMS[a][2]]
    if t_arms:
        rows = []
        for a in arms:
            rr = list(runs[a].values())
            steps = sum(r["tr"].get("train_steps", 0) for r in rr)
            rows.append([a, f"{steps:,}" if a in t_arms else "not counted (one per 5 actions)",
                         f"{steps / max(sum(r['taken'] for r in rr), 1):.2f}" if a in t_arms else "0.20 at most",
                         f"{st.median(r['ms_med'] for r in rr if r['ms_med'] is not None):.1f}",
                         f"{st.median(r['aps'] for r in rr):.0f}"])
        out += ["**T (`EVAL_TRAIN_EVERY`, `EVAL_TRAIN_RECENT`)**: `wall_ms_med` is the median over a run's moves, "
                "so for an arm that trains on one move in five it is a move without a training step.", ""]
        out += table(["Arm", "training steps", "steps per action", "median over runs of `wall_ms_med`",
                      "median act/s"], rows, right=(1, 2, 3, 4)) + [""]
    return out


def flag_problems(runs):
    """Runs whose run_config.json does not show their arm's flags."""
    bad = []
    for arm, per in runs.items():
        if arm not in S.ARMS:
            continue
        want = S.arm_env(arm, S.CAP)
        for g, r in per.items():
            c = r["cfg"]
            ok = (sorted(str(c.get("upgrades", "")).split(",")) == sorted(want["EVAL_UPGRADES"].split(","))
                  and c.get("label") == "novel" and c.get("return_map") is True and c.get("max_actions") == S.CAP
                  and c.get("seed_source") == "env" and c.get("reset_on_level") is True
                  and c.get("train_every", 5) == int(want.get("EVAL_TRAIN_EVERY", 5))
                  and c.get("train_recent", 0) == int(want.get("EVAL_TRAIN_RECENT", 0))
                  and c.get("train_frequency") == int(want.get("EVAL_TRAIN_EVERY", 5)))
            if not ok:
                bad.append(f"{arm} {g}")
    return bad


def write_csv(runs, path):
    cols = (["arm", "game", "tier", "levels", "levelup_actions", "levels_by_1000", "levels_by_2000", "unique_states",
             "actions_taken", "moves_recorded", "act_per_s", "wall_ms_med", "ended", "upgrades", "train_every",
             "train_recent", "train_steps", "map_moves", "untried_first"]
            + ["sched_pulls_" + n for n in SCHEDULES] + ["sched_decisions_" + n for n in SCHEDULES]
            + ["gate_walk", "gate_stay", "objclick_redrawn", "objclick_kept", "objclick_mean_shapes",
               "run_folder", "commit"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for arm in [a for a in S.TABLE_ORDER if a in runs]:
            for g in [g for g in S.game_order() if g in runs[arm]]:
                r = runs[arm][g]
                sched = "S" in S.ARMS[arm][2]
                w.writerow([arm, g, tiers.tier_of(g), r["levels"], ";".join(map(str, r["ups"])), level_by(r, 1000),
                            level_by(r, 2000), r["unique"], r["taken"], r["n"], r["aps"], r["ms_med"], r["end"],
                            r["cfg"].get("upgrades", ""), r["cfg"].get("train_every", 5), r["cfg"].get("train_recent", 0),
                            r["tr"].get("train_steps", ""), r["mp"].get("map_actions", ""), r["mp"].get("untried_first", "")]
                           + [r["mp"].get("sched_pulls_" + n, "") for n in SCHEDULES]
                           + [r["mp"].get("sched_decisions_" + n, "") for n in SCHEDULES]
                           + ["" if sched else r["mp"].get("gate_walk", ""), "" if sched else r["mp"].get("gate_stay", ""),
                              r["up"].get("objclick_redrawn", ""), r["up"].get("objclick_kept", ""),
                              r["up"].get("objclick_mean_shapes", ""), r["dir"], r["commit"][:10]])


def write(out, docs=False):
    """Write <out>/report.md and <out>/runs.csv; return the short version (the answer and the arm table)."""
    runs = load(out)
    meta = read_json(os.path.relpath(out, ROOT), "sweep.json")
    jobs = S.sweep_jobs()
    arms = [a for a in S.TABLE_ORDER if a in runs]
    n_done = sum(len(v) for v in runs.values())
    head = [f"# Upgrade screen 3: the tables", "",
            f"*Generated by `experiments/goose/screen3_report.py` from `{os.path.relpath(out, ROOT)}/manifest.tsv`. "
            f"Exploratory: one seed, {S.CAP:,} actions a game. The rule these tables are read by is in "
            f"`docs/1-goose-upgrades/upgrade-screen3.md`.*", "",
            f"Sweep started {meta.get('started', '?')} from commit `{str(meta.get('commit', '?'))[:7]}`"
            f"{' (+ uncommitted changes)' if meta.get('dirty') else ''}, {meta.get('jobs', '?')} runs at a time, "
            f"engine mode: {'`--offline`' if meta.get('offline') else 'default (online scorecard)'}.",
            f"Runs in the manifest: **{n_done} of {len(jobs)}**."]
    missing = [f"{j.arm} {j.game}" for j in jobs if j.game not in runs.get(j.arm, {})]
    if "base" not in runs:
        text = "\n".join(head + ["", "No `base` run is in the manifest yet: nothing to compare."])
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write(text + "\n")
        return text
    games = [g for g in S.game_order() if all(g in runs[a] for a in arms)]
    partial = len(games) < len({j.game for j in jobs}) or len(arms) < len({j.arm for j in jobs})
    head += [f"Games on which every arm that has runs is finished: **{len(games)} of {len({j.game for j in jobs})}**"
             + (f" ({', '.join(games)})" if partial else "") + ". Every comparison below is on these games.", ""]
    if not games:
        text = "\n".join(head + ["No game has every arm yet: nothing to compare."])
        with open(os.path.join(out, "report.md"), "w") as f:
            f.write(text + "\n")
        return text

    base, base2 = total(runs, "base", games), (total(runs, "base2", games) if "base2" in runs else None)
    order = rank(runs, games)
    vs = {a: against_base(runs, a, games) for a in arms if a != "base"}
    noise = {a: clears_noise(total(runs, a, games), base, base2, vs[a]["lo"]) for a in order}
    effects = {u: main_effect(runs, u, games) for u in "SOT"}
    have_effects = all(effects.values())

    # ---- the answer
    ans = ["## The answer", ""]
    if partial:
        ans += [f"**PARTIAL SWEEP: {n_done} of {len(jobs)} runs.** What follows is on {len(games)} games"
                + (f" and the arms {', '.join(arms)}" if len(arms) < len({j.arm for j in jobs}) else "") + ".", ""]
    if len(games) == 25 and not ANCHOR[0] <= base <= ANCHOR[1]:
        ans += [f"**CHECK THE SETUP: `base` completed {base} levels, outside the sanity anchor of "
                f"{ANCHOR[0]} to {ANCHOR[1]} (near 15 was expected).**", ""]
    if order:
        top = order[0]
        tie = [a for a in order[1:] if total(runs, a, games) == total(runs, top, games)]
        ans += [f"- **Ranked first of the upgraded arms: `{top}`**, {total(runs, top, games)} levels by {S.CAP:,} actions "
                f"against `base` {base}" + (f" and `base2` {base2}" if base2 is not None else " (no `base2` yet)")
                + f"; difference from `base` {vs[top]['delta']:+d}, 95% interval [{vs[top]['lo']:+d}, {vs[top]['hi']:+d}]: "
                f"**{'clears the noise' if noise[top] else 'inside the noise'}**."
                + (f" Tied on score with {', '.join('`%s`' % a for a in tie)}; the tie-breaks put `{top}` first." if tie else "")]
        won = [a for a in order if noise[a]]
        ans += ["- Arms that clear the noise: " + (", ".join(f"`{a}`" for a in won) if won else "**none**") + "."]
    if have_effects:
        ans += ["- Main effects on the score (the four arms with an upgrade minus the four without): "
                + "; ".join(f"{u} {effects[u]['effect']:+.2f} [{effects[u]['lo']:+.2f}, {effects[u]['hi']:+.2f}]" for u in "SOT")
                + "." + (f" They {'agree' if effects_agree(S.ARMS[order[0]][2], effects) else 'do not agree'} "
                         f"with `{order[0]}` as the best combination." if order else "")]
    if base2 is not None:
        b = vs["base2"]
        ans += [f"- Chance: `base2`, the same agent as `base`, completed {base2} levels against {base} "
                f"({b['delta']:+d}; better / same / worse on {b['better']} / {b['same']} / {b['worse']} games)."]
    if len(games) == 25 and ANCHOR[0] <= base <= ANCHOR[1]:
        ans += [f"- Sanity anchor: `base` completed {base} levels (near 15 expected, {ANCHOR[0]} to {ANCHOR[1]} accepted): fine."]

    # ---- per arm
    rows = []
    for a in ["base"] + order + (["base2"] if "base2" in runs else []):
        v = vs.get(a)
        rows.append([f"`{a}`", order.index(a) + 1 if a in order else "-",
                     total(runs, a, games, 1000), total(runs, a, games, 2000), f"**{total(runs, a, games)}**",
                     f"{v['delta']:+d}" if v else "-", f"[{v['lo']:+d}, {v['hi']:+d}]" if v else "-",
                     f"{v['better']} / {v['same']} / {v['worse']}" if v else "-",
                     ("clears" if noise[a] else "inside") if a in order else "-",
                     f"{sum(runs[a][g]['unique'] for g in games):,}",
                     f"{st.median(r['aps'] for r in runs[a].values()):.0f}", f"{len(runs[a])} of {len({j.game for j in jobs})}"])
    per_arm = table(["Arm", "Rank", "Levels by 1,000", "by 2,000", f"by {S.CAP:,} (score)", "vs `base`", "95% interval",
                     "Games better / same / worse", "Noise", "Unique states", "Median act/s", "Runs finished"],
                    rows, right=(1, 2, 3, 4, 5, 9, 10))

    # ---- by tier, game by arm
    rows = [[f"`{a}`"] + [sum(runs[a][g]["levels"] for g in games if g in tier) for _, tier in TIER_GROUPS]
            + [total(runs, a, games)] for a in arms]
    by_tier = table(["Arm"] + [f"{name} ({sum(g in tier for g in games)} games)" for name, tier in TIER_GROUPS] + ["all"],
                    rows, right=(1, 2, 3, 4))
    rows = []
    for name, tier in TIER_GROUPS:
        for g in [g for g in S.game_order() if g in tier and any(g in runs[a] for a in arms)]:
            rows.append([g, name] + [runs[a][g]["levels"] if g in runs[a] else "" for a in arms])
    by_game = table(["Game", "Tier"] + [f"`{a}`" for a in arms], rows, right=tuple(range(2, 2 + len(arms))))

    # ---- main effects
    eff = []
    if have_effects:
        rows = []
        for u, name in (("S", "S `map_sched`"), ("O", "O `objclick`"), ("T", "T train more")):
            e = effects[u]
            by = [main_effect(runs, u, games, t) for t in BUDGETS[:2]]
            rows.append([name, " ".join(e["yes"]), " ".join(e["no"]),
                         f"{st.mean(total(runs, a, games) for a in e['yes']):.2f}",
                         f"{st.mean(total(runs, a, games) for a in e['no']):.2f}",
                         f"**{e['effect']:+.2f}**", f"[{e['lo']:+.2f}, {e['hi']:+.2f}]"]
                        + [f"{b['effect']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]" for b in by])
        eff = table(["Upgrade", "arms with it", "arms without", "mean score with", "mean score without",
                     "main effect", "95% interval", "effect by 1,000 actions", "effect by 2,000 actions"], rows,
                    right=(3, 4, 5))
        eff += ["", "The same, pair by pair: the levels an upgrade adds to each of the four arms without it "
                "(the main effect is the mean of the four).", ""]
        eff += table(["Upgrade", "added to", "", "", ""],
                     [[name] + [f"`{a}` to `{b}`: {d:+d}" for a, b, d in added_to(runs, u, games)]
                      for u, name in (("S", "S `map_sched`"), ("O", "O `objclick`"), ("T", "T train more"))])
    else:
        eff = ["Not all eight arms have runs yet: no main effects."]

    body = head + ans + ["", "## Per arm", "",
                         "Ranked arms first to last between `base` and `base2`. \"vs `base`\" and its interval are on the "
                         "score; the interval resamples games 10,000 times (seed 0), as `tools/paired_compare.py` does. "
                         "Unique states: summed over the games. Median act/s: over the arm's runs, 4 runs sharing the GPU.",
                         ""] + per_arm
    body += ["", "## By tier", "", f"Levels by {S.CAP:,} actions.", ""] + by_tier
    body += ["", "## Game by arm", "", f"Levels by {S.CAP:,} actions.", ""] + by_game
    body += ["", "## Main effects", ""] + eff
    others = [a for a in arms if a != "base"]
    body += ["", "## Time to level, each arm against `base`", "",
             "From `tools/levels_by_budget.py`'s functions (budgets 1,000, 2,000 and 5,000). A level race is one level "
             "both runs started and at least one finished: the arm that took fewer actions on it wins. `base2`'s row "
             "is what chance alone does.", ""] + time_to_level(runs, games, others)
    if base2 is not None:
        b = vs["base2"]
        body += ["", "## `base` against `base2`: the size of chance", "",
                 f"The two are the same agent with the same seed; they differ only because GPU runs stop being "
                 f"identical after a few hundred actions. So their gap is a LOWER bound on chance: a different seed "
                 f"would add more. Score {base} against {base2} ({b['delta']:+d}, 95% interval [{b['lo']:+d}, {b['hi']:+d}]); "
                 f"`base2` better / same / worse on {b['better']} / {b['same']} / {b['worse']} games; "
                 f"levels by 1,000 actions {total(runs, 'base', games, 1000)} against {total(runs, 'base2', games, 1000)}, "
                 f"by 2,000 {total(runs, 'base', games, 2000)} against {total(runs, 'base2', games, 2000)}. "
                 f"Its level races are the `base2` row of the table above.",
                 "", "Games where the two differ: "
                 + (", ".join(f"{g} ({runs['base'][g]['levels']} vs {runs['base2'][g]['levels']})" for g in games
                              if runs["base"][g]["levels"] != runs["base2"][g]["levels"]) or "none") + ".",
                 "", "Does it matter which of the two runs an arm is compared with? Not part of the rule, which names "
                 "`base`; shown because every arm above is compared with the same one run.", ""]
        rows = []
        for a in order:
            b2, (both, lo, hi) = against_base(runs, a, games, base="base2"), against_both(runs, a, games)
            rows.append([f"`{a}`", f"{vs[a]['delta']:+d} [{vs[a]['lo']:+d}, {vs[a]['hi']:+d}]",
                         f"{vs[a]['better']} / {vs[a]['same']} / {vs[a]['worse']}",
                         f"{b2['delta']:+d} [{b2['lo']:+d}, {b2['hi']:+d}]", f"{b2['better']} / {b2['same']} / {b2['worse']}",
                         f"{both:+.1f} [{lo:+.1f}, {hi:+.1f}]"])
        body += table(["Arm", "vs `base` [95% interval]", "games better / same / worse", "vs `base2` [95% interval]",
                       "games better / same / worse", "vs the mean of the two [95% interval]"], rows)
    body += ["", "## What each upgrade did", ""] + upgrade_tables(runs, arms)
    fails = [r for r in S.read_tsv(out, "failed.tsv")]
    still = [r for r in fails if r[0] not in runs.get(r[1], {})]
    bad_flags = flag_problems(runs)
    commits = sorted({r["commit"][:7] for per in runs.values() for r in per.values()})
    early = [f"{a} {g} ({r['taken']:,} actions, {r['end']})" for a in arms for g, r in runs[a].items() if r["taken"] < S.CAP]
    body += ["## Runs", "",
             f"- In the manifest: {n_done} of {len(jobs)}. Missing: " + (", ".join(missing) if missing else "none") + ".",
             f"- Failures logged: {len(fails)}" + (f", of which {len(fails) - len(still)} succeeded on a later try" if fails else "")
             + (". Still failed: " + "; ".join(f"{r[1]} {r[0]}: {r[3]}" for r in still) if still else "") + ".",
             "- Flags in each run's `run_config.json`: " + ("as the arm table says, on every run." if not bad_flags
                                                           else "**NOT as the arm table says on: " + ", ".join(bad_flags) + "**"),
             "- Runs that ended before the cap: " + (", ".join(early) if early else "none") + ".",
             f"- Commit recorded by the runs: {', '.join('`%s`' % c for c in commits)}.",
             f"- Recordings: `{os.path.relpath(out, ROOT)}/runs/<arm>/<game>_s{S.SEED}/runs/<time>/<game>/` "
             f"(`transitions/`, `metrics.json`, `run_config.json`, the stats files). One row per run: "
             f"`{os.path.relpath(out, ROOT)}/runs.csv`."]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(body) + "\n")
    write_csv(runs, os.path.join(out, "runs.csv"))
    if docs:
        write_csv(runs, os.path.join(ROOT, DOCS_CSV))
    return "\n".join(ans[2:] + [""] + per_arm + ["", f"tables: {os.path.relpath(out, ROOT)}/report.md   "
                                                  f"one row per run: {os.path.relpath(out, ROOT)}/runs.csv"
                                                  + (f" and {DOCS_CSV}" if docs else "")])
