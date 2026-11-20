#!/usr/bin/env python3
"""explorer_probe.py - which of the groupmate's agents for keyboard games helps behind this project's Goose?

The problem it fixes: the planner (E20) watches clicks only, so it does nothing on a game played with
keys. The groupmate's report has three further agents for those games: E21 (an explorer that searches
the screens systematically and carries what a level taught), E22 (E21 plus a movement model, goal rules
and a planner) and E24 (E21 with another way of exploring). In their own test these had a weak explorer
for level 1. This asks, from recordings and on the CPU, what each does once it has Goose's level 1.

How. A recording of Goose is replayed on the local engine. While level 1 lasts the agent is handed the
recording's moves in the one place where it chooses a move (E21's `_pick`), so it books and learns from
each as if it had chosen it. From the first move of level 2 it plays alone, as the groupmate built it
(its planner if it has one, its explorer, then its own small Goose), to the recording's action 5,000.
Beside each row stands what the recording's Goose went on to do in the same actions.

    uv run python experiments/kp/explorer_probe.py                      # E21 after Goose's level 1
    uv run python experiments/kp/explorer_probe.py --agents E21b        # E21 with nothing carried (its own ablation)
    uv run python experiments/kp/explorer_probe.py --agents E22,E24 --ref <folder>    # needs their two files
    uv run python experiments/kp/explorer_probe.py --from-start         # no recorded move is fed: no Goose level 1
    uv run python experiments/kp/explorer_probe.py --games vc33 --force-keys          # E21 on a game with clicks only

E21 is custom_agents/kp_ref/e21_object_explorer.py. E22 and E24 are not in the repo (they lost here):
--ref names a folder holding e22_keyboard_planner.py and e24_go_explore.py as they are in the groupmate's
archive. Recordings and games are run_reference.py's kind: dev and seen games only unless --registered.
Output: <out>/<tsv> (one row per recording and agent) and <out>/runs/. Exploratory: it gives no verdict.
The agent this led to is the bridge's `[explore] keys = "e21"` (custom_agents/kp_bridge.py), and
tools/kp_offline.py takeover --config experiments/kp/kp_e21.toml is the same check through that bridge.
Write-up: docs/4-knowledge-planner/kp-explorer.md.
"""
import argparse
import collections
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "custom_agents"), os.path.join(ROOT, "experiments", "kp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kp_bridge  # noqa: E402  (the one loader of the reference's files)
import run_reference as R  # noqa: E402  (recordings, the engine set-up, the worker pool)

OUT = "results/kp/explorer_probe"
GAMES = "cd82,dc22,m0r0,g50t,ls20,tu93,tr87"        # the dev and seen games played with keys
RUNS = ["results/kp/test/20261119_181131/manifest.tsv:SOT", "results/kp/test/20261119_181131/manifest.tsv:SOT2",
        "results/kp/dev/20261119_162448/manifest.tsv:SOT", "results/screen3/20261116_235041/manifest.tsv:SOT"]
ELSEWHERE = {"E22": ("e22_keyboard_planner", "KeyboardPlannerAgent"), "E24": ("e24_go_explore", "GoExploreAgent")}


def agent_class(kind, ref_dir):
    """(E18's module, E21's module, the class to build, its extra arguments) for E21, E21b, E22 or E24."""
    e18, e21 = kp_bridge.load_reference()[0], kp_bridge.load_explorer()
    if kind in ELSEWHERE:
        if not ref_dir:
            raise SystemExit(f"{kind} is not in the repo: --ref <the folder with its file>")
        if ref_dir not in sys.path:
            sys.path.insert(0, ref_dir)
        module, name = ELSEWHERE[kind]
        return e18, e21, getattr(kp_bridge.load(module)[0], name), {}
    return e18, e21, e21.ObjectExplorerAgent, ({"transfer": False} if kind == "E21b" else {})


def kind_at(e18, e21, grid, bg, x, y):
    """E21's name for the piece under a click (it books clicks by the kind of piece)."""
    colour = int(grid[y, x])
    if colour not in bg:
        for ys, xs in e18._components(grid == colour):
            if ((ys == y) & (xs == x)).any():
                return e21.Part(ys, xs, colour).kind
    return "background"


def goose_level_ups(run_dir, cap):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        return [int(a) for a, _ in (json.load(f).get("levelup_events") or []) if a <= cap]


def probe_one(task):
    run, cap, out, ref_dir, kind, from_start, force_keys = task
    t0 = time.time()
    tag = f"{run.label}_{run.arm}_{run.game}_s{run.seed}_{kind}" + ("_fromstart" if from_start else "")
    res = dict(game=run.game, seed=run.seed, arm=run.arm, label=run.label, agent=kind, goose_ups="", goose_levels="",
               levels="", ups="", plan_steps="", explore_steps="", own_goose_steps="", cleared_by="", top_rule="",
               controlled="", moves="", stopped="", seconds=0.0)
    try:
        e18, e21, cls, extra = agent_class(kind, ref_dir)
        rec = R.read_recording(run.run_dir, cap)
        ups_goose = goose_level_ups(run.run_dir, cap)
        res.update(goose_ups=" ".join(map(str, ups_goose)), goose_levels=len(ups_goose))
        later = np.nonzero(rec["levels"] >= 1)[0]
        if from_start:                              # no recorded move is fed: the agent plays from the first move
            last, feed = None, iter(())
        elif not len(later):
            res["stopped"] = f"no prefix: no level 1 inside {cap:,} actions"
            return res
        else:
            last = int(later[0]) - 1                # the level-completing move: the last row recorded on level 1
            feed = iter(range(last + 1))
        offset = int(rec["action_nums"][0]) - 1     # the recording's first move is numbered 1 (2 after a starting RESET)
        if force_keys:                              # a game with clicks only, given to E21's own path (it would call E20)
            base = cls

            class cls(base):
                def reset(self):
                    base.reset(self)
                    self.click_only = False
        agent = cls(seed=0, knowledge_path=os.path.join(out, "runs", tag, "store.json"), **extra)
        steps = collections.defaultdict(collections.Counter)      # levels completed so far -> who chose the move
        last_mode, after_level1 = {}, {}

        def pick(node, parts, grid, legal):
            """In E21's place where a move is chosen: the recording's on level 1, the agent's own afterwards."""
            i = next(feed, None)
            if i is not None:
                if not np.array_equal(grid, rec["frames"][i]):
                    raise R.Mismatch(int(rec["action_nums"][i]))
                index = int(rec["actions"][i])
                if index < 5:
                    return f"ACTION{index + 1}"
                y, x = divmod(index - 5, R.GRID)
                return f"click:{kind_at(e18, e21, grid, agent.bg, x, y)}@{x},{y}"
            if not after_level1 and agent.level >= 1:       # what level 1 taught, before the first move of level 2
                after_level1.update(agent.xk.to_json())
            planned = agent.x_log["plan"]
            akey = cls._pick(agent, node, parts, grid, legal)
            mode = "plan" if agent.x_log["plan"] > planned else ("explore" if akey is not None else "own_goose")
            steps[agent.level][mode] += 1
            last_mode[agent.level] = mode
            return akey

        agent._pick = pick
        arc = R.Wrapped(R.arcade(), R.CountingEnv)
        try:
            row = e18.run_episode(arc, run.game, agent, None, seed=R.game_seed(run.run_dir), max_steps=cap - offset)
            res["stopped"] = "won the game" if row["state"] == "WIN" else f"the cap, on level {row['levels'] + 1}"
        except R.Mismatch as e:
            res["stopped"] = f"REPLAY MISMATCH before action {e}"
        ups = [a + offset for _, a in arc.env.level_ups]
        # metrics.json numbers a level-up one later than the completing move's own number in the recording:
        # the replay is put on metrics.json's clock, so the two lists can be read side by side
        if from_start:
            ups = [u + 1 for u in ups]
        elif ups and ups_goose and ups[0] == int(rec["action_nums"][last]) and ups_goose[0] - ups[0] in (0, 1):
            ups = [u + ups_goose[0] - ups[0] for u in ups]
        else:
            res["stopped"] += f" (level 1 replayed at {ups[:1]}, recorded at {ups_goose[:1]})"
        counted = [lv for lv in steps if lv >= 1 or from_start]
        rules = after_level1.get("KR_goal_rules_ranked", [])
        res.update(levels=arc.env.levels, ups=" ".join(map(str, ups)),
                   plan_steps=sum(steps[lv]["plan"] for lv in counted),
                   explore_steps=sum(steps[lv]["explore"] for lv in counted),
                   own_goose_steps=sum(steps[lv]["own_goose"] for lv in counted),
                   cleared_by=" ".join(f"L{lv + 1}:{last_mode.get(lv, '?')}"
                                       for lv in range(0 if from_start else 1, arc.env.levels)),
                   top_rule=rules[0]["meaning"] if rules else "", controlled=after_level1.get("KM_controlled_kind") or "",
                   moves=sum(1 for v in after_level1.get("KM_displacement", {}).values() if v))
        os.makedirs(os.path.join(out, "runs", tag), exist_ok=True)
        with open(os.path.join(out, "runs", tag, "log.json"), "w") as f:
            json.dump(dict(result=res, after_level1=after_level1, at_end=agent.xk.to_json(),
                           steps={str(k): dict(v) for k, v in steps.items()},
                           k_log=[str(x) for x in getattr(agent, "k_log", [])][:400]), f, indent=1, default=str)
    except Exception:
        res["stopped"] = "ERROR: " + traceback.format_exc().strip().splitlines()[-1]
        os.makedirs(os.path.join(out, "runs", tag), exist_ok=True)
        with open(os.path.join(out, "runs", tag, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
    res["seconds"] = round(time.time() - t0, 1)
    return res


def table(done, agents):
    """Levels by the cap, summed over the recordings that both sides played: the recording's Goose, then each agent."""
    rows = [r for r in done if r["levels"] != ""]
    key = lambda r: (r["game"], r["label"], r["arm"], r["seed"])
    lines = ["| Game | Recordings | Goose | " + " | ".join(agents) + " |", "|---|---|---|" + "---|" * len(agents)]
    games = sorted({r["game"] for r in rows})
    for g in games + ["all"]:
        mine = [r for r in rows if g in (r["game"], "all")]
        n = len({key(r) for r in mine})
        goose = sum(int(r["goose_levels"]) for r in mine if r["agent"] == agents[0])
        cells = [sum(int(r["levels"]) for r in mine if r["agent"] == a) for a in agents]
        lines.append(f"| {g} | {n} | {goose} | " + " | ".join(map(str, cells)) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agents", default="E21", help="from E21, E21b, E22, E24 (comma list)")
    ap.add_argument("--ref", default="", help="the folder with E22's and E24's files (they are not in the repo)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tsv", default="probe.tsv")
    ap.add_argument("--games", default=GAMES)
    ap.add_argument("--runs", nargs="*", default=RUNS, help="<manifest>[:<arm>] ...")
    ap.add_argument("--registered", action="store_true", help="allow games outside dev and seen (a registered test only)")
    ap.add_argument("--actions", type=int, default=R.CAP)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--from-start", action="store_true", help="feed no recorded move: the agent plays from the first move")
    ap.add_argument("--force-keys", action="store_true", help="give a game with clicks only to E21's own path")
    a = ap.parse_args()
    os.chdir(ROOT)
    out, agents = os.path.join(ROOT, a.out), R.split(a.agents)
    runs = R.read_runs(a.runs, R.split(a.games), a.registered)
    tasks = [(r, a.actions, out, os.path.abspath(a.ref) if a.ref else "", k, a.from_start, a.force_keys)
             for k in agents for r in runs]
    print(f"{len(runs)} recordings x {a.agents} on {a.games}; {a.jobs} at a time")
    done = R.run_all(probe_one, tasks, a.jobs,
                     lambda r: f"{r['agent']} {r['game']} s{r['seed']} {r['arm']}: levels {r['levels']} (Goose "
                               f"{r['goose_levels']}), planned steps {r['plan_steps']}, {r['stopped']}")
    done.sort(key=lambda r: (agents.index(r["agent"]), r["game"], str(r["label"]), r["arm"], str(r["seed"])))
    R.write_tsv(os.path.join(out, a.tsv), done)
    print("\n".join(table(done, agents)))
    print(f"wrote {os.path.join(a.out, a.tsv)}")


if __name__ == "__main__":
    main()
