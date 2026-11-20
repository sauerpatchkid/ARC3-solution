#!/usr/bin/env python3
"""kp_offline.py - where the Knowledge Planner works and where it does not, from recordings, on the CPU.

The problem it fixes: a GPU run says how many levels the planner's arm ends
with, and not why. Which games let the planner see a level as slots, learn a
rule from the solved board, build a model and find a plan? A finished Goose
recording can answer that with no new run: it replays on the engine move for
move (docs/4-knowledge-planner/kp-phase0.md), so the planner can watch it, or
take over from it. Plan: docs/4-knowledge-planner/kp-plan.md, Phase 2.

    uv run python tools/kp_offline.py induct   [--runs ...] [--games a,b]
        the planner watching a whole recording and never taking a move: one row
        per cleared level with what it could learn there
    uv run python tools/kp_offline.py takeover [--runs ...] [--games a,b]
        the recording plays level 1, then the planner plays alone to action
        5,000: one row per recording, beside what the recording's Goose had
    uv run python tools/kp_offline.py funnel   [--episodes FOLDER ...]
        the per-game funnel of the plan's section 7.1, from the files a
        takeover (or a real run with EVAL_KP=1) wrote

Every episode drives the real bridge (custom_agents/kp_bridge.py) with the
calls action.py's hook makes, on the local engine, with the recording standing
in for Goose. So what is measured is the component the test runs, and it
writes the files a real run writes (kp_stats.json, events.jsonl, store.json).
That the bridge makes the reference's own clicks is checked separately
(run_reference.py prefix --whole, and tests/test_kp_bridge.py).

--runs, --games, --registered and the default list of recordings are
run_reference.py's: every unguided arm recorded since 6 Oct 2026, dev and seen
games only. Games that never offer a click are left out: the planner is
dormant there. Each episode runs in its own process (PYTHONHASHSEED=0, one
thread, no GPU). Output: results/kp/phase2/, or --out.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "custom_agents"), os.path.join(ROOT, "experiments", "kp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kp_bridge  # noqa: E402
import run_reference as R  # noqa: E402  (the recordings, the engine set-up and the worker pool of Phase 0)
from canon import OnlineCanonicalizer  # noqa: E402
from upgrades import BarDetector  # noqa: E402

OUT = "results/kp/phase2"
NO_CLICKS = ("g50t", "ls20", "re86", "tr87", "tu93", "wa30")       # kp-phase0.md, section 5
STEPS = ("entered", "slots held", "had a rule", "model settled", "plan found", "cleared by planning")


def unpack(index):
    """The agent's index as the triple Goose's sampler returns."""
    return (index, None, None) if index < 5 else (5, divmod(index - 5, R.GRID), index - 5)


def tag_of(run):
    return f"{run.label}_{run.arm}_{run.game}_s{run.seed}"


# ---- one episode: the bridge, the engine, and a recording in Goose's place -----------------------------
def play(task):
    """`mode` "induct": every recorded move is Goose's and the planner only watches. "takeover": the
    recording's moves run out at its first level-up, and the episode stops when the planner asks
    Goose for a move. Returns the recording's row and one row per level cleared."""
    run, mode, cap, out, config = task
    t0 = time.time()
    rec = R.read_recording(run.run_dir, cap)
    row = dict(run._asdict(), first_level_up="", goose_levels=int(rec["levels"].max(initial=0)), levels=0,
               level_actions="", stopped="", errors=0, seconds=0.0)
    feed = len(rec["action_nums"])
    if mode == "takeover":
        later = np.nonzero(rec["levels"] >= 1)[0]
        if not len(later):
            return dict(row, levels="", stopped=f"no prefix: no level 1 inside {cap:,} actions"), []
        feed = int(later[0])                    # the moves up to and with the level-completing one
        row["first_level_up"] = int(rec["action_nums"][feed - 1])
    folder = os.path.join(out, mode, tag_of(run))
    settings = kp_bridge.load_settings(config)
    b = kp_bridge.Bridge(folder, settings, plan=mode == "takeover", game_id=run.game)
    e18, env = b.e18, R.arcade().make(run.game, seed=R.game_seed(run.run_dir))
    obs, counter, i, ups, clears = env.reset(), 0, 0, [], []
    # With `[see] mask = "agent"`: Goose's decoration mask, built move by move as the agent builds it
    # (the shared detector, then the `bars` upgrade's cells, which are one move behind at the hook).
    canon, bars, prev = OnlineCanonicalizer(), BarDetector(), None
    while counter < cap:
        if obs.state.name == "WIN":
            row["stopped"] = "won the game"
            break
        if obs.state.name == "GAME_OVER":
            b.on_game_over()
            obs, counter, prev = env.reset(), counter + 1, None
            continue
        mask, screen = None, R.last_frame(obs).astype(np.uint8)
        if settings["see"]["mask"] == "agent":
            if prev is not None:
                canon.update(prev, screen)
            mask = canon.mask | bars.sticky
            if prev is not None:
                bars.update(prev, screen)
        b.step(obs.frame, obs.levels_completed, env.action_space, mask, counter)
        move = int(rec["actions"][i]) if i < feed else None
        if move is not None and not np.array_equal(R.last_frame(obs), rec["frames"][i]):
            row["stopped"] = f"REPLAY MISMATCH before action {int(rec['action_nums'][i])}"
            break
        seen = len(b.levels)
        _, _, coord = b.override(*unpack(move if move is not None else 0))
        if len(b.levels) > seen and "learned" in b.levels[-1]:         # a level was cleared, and the planner learned from it
            done = b.levels[-1]
            clears.append(dict(run._asdict(), level=done["level"], actions=done["left_at"] - done["entered_at"],
                               screens=done["screens"], screens_that_fit=done["screens_that_fit"],
                               cleared=done["ended"], **done["learned"]))
        if not b.on:
            row["stopped"] = "an error inside the planner (see events.jsonl)"
            break
        if b.took:
            final = 5 + coord
        elif move is None:
            why = b.level["gave_up"] or kp_bridge.why_fell_back(e18, b.agent, b.grid)
            row["stopped"] = f"gave up on level {b.level['level']}: {why}"
            break
        else:
            final, i = move, i + 1
        b.record(final)
        action, data = R.engine_move(final)
        level = obs.levels_completed
        obs, counter = env.step(action, data=data), counter + 1
        prev = screen if obs.levels_completed == level else None        # no transition is learned across a level-up
        if obs.levels_completed > level:
            ups.append(counter)
        if mode == "induct" and i == feed:
            row["stopped"] = "the recording ended"
            break
    else:
        row["stopped"] = f"the cap, on level {obs.levels_completed + 1}"
    b.dump(os.path.join(folder, "kp_stats.json"), won=obs.state.name == "WIN")
    row.update(levels=int(obs.levels_completed), errors=b.n["kp_errors"], seconds=round(time.time() - t0, 1),
               level_actions=" ".join(str(q - p) for p, q in zip(ups, ups[1:])))
    return row, clears


# ---- the commands ---------------------------------------------------------------------------------------
def click_games(runs):
    return [r for r in runs if r.game not in NO_CLICKS]


def episodes(a, mode):
    runs = click_games(R.read_runs(a.runs, R.split(a.games), a.registered))
    say = lambda r: (f"{r[0]['label']} {r[0]['arm']} {r[0]['game']} s{r[0]['seed']}: {r[0]['levels']} levels, "
                     f"{r[0]['stopped']} ({r[0]['seconds']:.0f} s)")
    done = R.run_all(play, [(r, mode, a.actions, a.out, a.config) for r in runs], a.jobs, say)
    rows = sorted((r for r, _ in done), key=lambda r: (r["game"], r["label"], r["arm"], r["seed"]))
    clears = sorted((c for _, cs in done for c in cs), key=lambda c: (c["game"], c["label"], c["arm"], c["seed"], c["level"]))
    return rows, clears


def induct(a):
    rows, clears = episodes(a, "induct")
    R.write_tsv(os.path.join(a.out, "induct_runs.tsv"), rows)
    R.write_tsv(os.path.join(a.out, "induct.tsv"), clears)
    bad = [r for r in rows if r["errors"] or "MISMATCH" in r["stopped"]]
    lines = ["| Game | Level | Clears seen | Screens that fit the slots | Slots (dynamic) | Click effect | "
             "Solved board fits | Negatives | Clears with a consistent rule | Rule carried in: confirmed again / "
             "refuted / no evidence | The top rule after it |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in sorted({c["game"] for c in clears}):
        for level in sorted({c["level"] for c in clears if c["game"] == g}):
            cs = [c for c in clears if c["game"] == g and c["level"] == level]
            share = sum(c["screens_that_fit"] for c in cs) / max(1, sum(c["screens"] for c in cs))
            span = lambda k: (lambda v: str(v[0]) if v[0] == v[-1] else f"{v[0]} to {v[-1]}")(sorted(c[k] for c in cs))
            kinds = sorted({f"{c['dynamics']}{', cycle ' + str(c['cycle']) if c['cycle'] not in ('', None) else ''}"
                            for c in cs if c["dynamics"]})
            tops = sorted({c["top_rule_says"] for c in cs if c["top_rule_says"]})
            lines.append(R.md([g, level, len(cs), f"{share:.0%}", f"{span('slots')} ({span('dynamic_slots')})",
                               "; ".join(kinds) or "none seen", f"{sum(c['solved_board_fits'] for c in cs)} of {len(cs)}",
                               span("negatives"), f"{sum(c['rules_consistent'] > 0 for c in cs)} of {len(cs)}",
                               " / ".join(str(sum(c["carried_rule"] == k for c in cs))
                                          for k in ("confirmed again", "refuted", "no evidence")),
                               "; ".join(tops) or "none"]))
    lines += ["", f"{len(rows)} recordings watched, {len(clears)} level clears. Recordings with an error in the planner "
              f"or a replay mismatch: {len(bad)}."]
    R.finish(os.path.join(a.out, "induct.md"), lines)


def takeover(a):
    rows, clears = episodes(a, "takeover")
    R.write_tsv(os.path.join(a.out, "takeover.tsv"), rows)
    R.write_tsv(os.path.join(a.out, "takeover_clears.tsv"), clears)
    lines = ["| Game | Prefixes | Levels by the cap: the planner (each prefix) | ... the recording's Goose | "
             "Prefixes where the planner cleared a level | How it stopped |", "|---|---|---|---|---|---|"]
    for g in sorted({r["game"] for r in rows}):
        rs = [r for r in rows if r["game"] == g and r["first_level_up"] != ""]
        count = lambda values: ", ".join(f"{v}: {n}" for v, n in sorted(_tally(values).items()))
        why = _tally(r["stopped"].split(": ", 1)[-1] if r["stopped"].startswith("gave up") else r["stopped"] for r in rs)
        lines.append(R.md([g, len(rs), count(r["levels"] for r in rs), count(r["goose_levels"] for r in rs),
                           f"{sum(r['levels'] > 1 for r in rs)} of {len(rs)}",
                           "; ".join(f"{k} ({n})" for k, n in sorted(why.items(), key=lambda kv: -kv[1]))]))
    none = sorted({r["game"] for r in rows} - {r["game"] for r in rows if r["first_level_up"] != ""})
    lines += ["", "Levels are written as `levels reached: number of prefixes`. A prefix is a recording up to its "
              f"first level-up; a recording without one inside {a.actions:,} actions is left out"
              + (f" (no prefix at all on: {', '.join(none)})." if none else ".")]
    R.finish(os.path.join(a.out, "takeover.md"), lines)
    a.episodes = [os.path.join(a.out, "takeover")]
    funnel(a)


def _tally(values):
    out = defaultdict(int)
    for v in values:
        out[v] += 1
    return out


def level_steps(stats, events):
    """The funnel's steps for every level after the first that a run entered with the planner awake.
    Each step counts only if the one before it does."""
    said = defaultdict(set)
    for e in events:
        if e["event"] == "reference":
            said[e["level"]].add(e["entry"][0])
    out = []
    for v in stats["levels"]:
        if v["level"] < 2:
            continue
        passed = [True, v["screens"] > 0 and v["screens_that_fit"] >= 0.95 * v["screens"],
                  "probe-set" in said[v["level"]],      # the reference had all three kinds of knowledge, and engaged
                  "goal" in said[v["level"]],           # probing and verifying over: it chose a rule to plan to
                  "plan" in said[v["level"]],           # a solver returned a plan to that rule
                  v["ended"] == "cleared by planning"]
        reached = [all(passed[:k + 1]) for k in range(len(passed))]
        out.append(dict(level=v["level"], reached=reached, ended=v["ended"], gave_up=v["gave_up"]))
    return out


def funnel(a):
    """Per game, over every episode folder under --episodes (a folder with kp_stats.json and events.jsonl)."""
    games = defaultdict(list)
    for root in a.episodes:
        for path in sorted(glob.glob(os.path.join(root, "**", "kp_stats.json"), recursive=True)):
            with open(path) as f:
                stats = json.load(f)
            with open(os.path.join(os.path.dirname(path), "events.jsonl")) as f:
                events = [json.loads(line) for line in f]
            if not stats["dormant"]:
                games[stats["game"]] += level_steps(stats, events)
    lines = [R.md(["Game", "Levels entered after the first"] + [s.capitalize() for s in STEPS[1:]]
                  + ["Cleared by Goose after a fallback", "Not cleared", "Where it stopped"]),
             R.md(["---"] * (len(STEPS) + 4))]
    for g in sorted(games):
        levels = games[g]
        if not levels:
            lines.append(R.md([g, 0] + [""] * (len(STEPS) + 2)))
            continue
        stops = _tally(v["gave_up"] for v in levels if v["gave_up"])
        lines.append(R.md([g] + [sum(v["reached"][k] for v in levels) for k in range(len(STEPS))]
                          + [sum(v["ended"] == "cleared by Goose" for v in levels),
                             sum(v["ended"] == "not cleared" for v in levels),
                             "; ".join(f"{k} ({n})" for k, n in sorted(stops.items(), key=lambda kv: -kv[1])) or "-"]))
    lines += ["", "Each step counts a level only if it passed every step before it. Slots held: at least 95% of the "
              "level's screens fit its first-screen slots. Had a rule: the planner had a clickable kind, a click "
              "effect and a confirmed rule, and engaged. Model settled: probing and verifying were over and it chose "
              "a rule to plan to. Plan found: a solver returned a plan. Cleared by planning: the level ended with the "
              "planner in charge."]
    R.finish(os.path.join(a.out, "funnel.md"), lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("induct", "takeover", "funnel"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=OUT)
        if name == "funnel":
            p.add_argument("--episodes", nargs="+", default=[os.path.join(OUT, "takeover")],
                           help="folders holding kp_stats.json and events.jsonl, at any depth")
            continue
        p.add_argument("--jobs", type=int, default=8, help="episodes at a time (default 8)")
        p.add_argument("--runs", nargs="*", default=[], help="<manifest>[:<arm>] or run folders (default: the plan's list)")
        p.add_argument("--games", default="", help="comma-separated (default: every dev and seen game with clicks)")
        p.add_argument("--actions", type=int, default=R.CAP, help=f"the budget, in the recording's action numbers ({R.CAP:,})")
        p.add_argument("--config", default=kp_bridge.DEFAULT_CONFIG, help="the planner's settings (default experiments/kp/kp.toml)")
        p.add_argument("--registered", action="store_true", help="allow games outside dev and seen (the test only)")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(a.out, exist_ok=True)
    {"induct": induct, "takeover": takeover, "funnel": funnel}[a.cmd](a)


if __name__ == "__main__":
    main()
