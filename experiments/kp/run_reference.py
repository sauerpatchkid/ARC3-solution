#!/usr/bin/env python3
"""run_reference.py - the groupmate's agents in their own harness, on this machine and after this project's Goose.

The problem it fixes: the Knowledge Planner (docs/4-knowledge-planner/kp-plan.md)
rests on three things nobody had checked here. Does the reference code
(custom_agents/kp_ref/, the groupmate's files, unedited) give the report's
numbers on this machine? Can a recorded Goose run be played again on the
engine, move for move? Does the reference's planner still work when this
project's Goose played level 1, and not its own small port? This answers all
three on the CPU, before anything is built.

    uv run python experiments/kp/run_reference.py repro
        E0, E18 and E20 in the reference's own run_episode, nothing replaced:
        its two games x seeds 0-5 x 1,500 actions, beside the report's table
    uv run python experiments/kp/run_reference.py replay-check [--runs <manifest>[:<arm>] ...] [--all]
        re-drive each recording on the engine: does every move give the recorded
        screen, and what does a level-completing move return?
    uv run python experiments/kp/run_reference.py prefix --games ft09,lp85 [--runs ...]
        the reference's E20 after a recorded Goose level 1: the recording
        stands in for the reference's explorer until its first level-up, then
        the planner plays alone, to 5,000 actions in all
    uv run python experiments/kp/run_reference.py prefix --whole --runs <run folder or manifest> ...
        the bridge against the reference (Phase 1): the recording of a run made
        with EVAL_KP=1 stands in for the explorer for the whole run. The
        reference in its own harness must take the moves the bridge took, with
        the same clicks, and leave to its explorer the moves the bridge left to Goose

Without --runs, the recordings are the plan's list (RUNS below): every
unguided arm recorded since 6 Oct 2026, when recordings began to hold the move
that ends an attempt. Only dev and seen games are read (tiers.py); the other
tiers need --registered, which belongs to the test.

Every episode runs in its own worker process, started with PYTHONHASHSEED=0,
one torch thread and no GPU, so a table comes out the same on every run.
Output: results/kp/phase0/ (one .tsv and one .md per command), or --out.

The reference is never edited, and it is loaded the way the bridge loads it
(custom_agents/kp_bridge.py, load_reference). `prefix` makes one replacement at
run time: StochasticGooseAgent.act, the reference's explorer, hands out the
recording's next move. `repro` and `replay-check` replace nothing. None of them
uses the bridge's own three replacements.
"""
import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import namedtuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "custom_agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import manifest as _manifest  # noqa: E402  (the repo's one manifest reader)
import tiers  # noqa: E402
from kp_bridge import load_reference, why_fell_back  # noqa: E402  (one loader for the bridge and for this tool)

GRID = 64
CAP = 5000                      # the test's budget: a prefix and the planner after it share it
OUT = "results/kp/phase0"

# The plan's "recordings to use" (Phase 0): label, manifest, arm (None = every arm in the manifest).
RUNS = [("screen3", "results/screen3/20261116_235041/manifest.tsv", None),     # nine arms, seed 0, 5,000 actions
        ("g4", "results/glassbox/g4/online/manifest.tsv", "A0-fresh"),          # the adopted Goose, seeds 0-2
        ("dev2", "results/glassbox/dev2/online/manifest.tsv", "A0-fresh"),      # seeds 3-6, dev games
        ("carry", "results/sweeps/sweep_20261112_135205.manifest", "A5")]       # the carry test, seeds 7-10
OBJCLICK = ("SOT", "O", "SO", "OT")     # screen 3's arms that choose clicks by shape; SOT is the test's base

# The report's Table 2 (the plan's Appendix D): levels per episode, seeds 0-5, 1,500 actions.
REPORT = {("ft09-0d8bbf25", "E0"): (1, 0, 1, 1, 1, 1), ("ft09-0d8bbf25", "E18"): (3, 0, 3, 3, 3, 3),
          ("ft09-0d8bbf25", "E20"): (6, 0, 6, 6, 6, 6), ("lp85-305b61c3", "E0"): (0, 2, 1, 1, 1, 1),
          ("lp85-305b61c3", "E18"): (0, 3, 3, 3, 3, 3), ("lp85-305b61c3", "E20"): (0, 4, 4, 5, 5, 4)}

# The go/no-go of Phase 0.4, fixed in the plan: game -> (levels to reach, share of prefixes, of the objclick ones).
GO = {"ft09": (5, 0.80, 3), "lp85": (4, 0.60, 2)}

Run = namedtuple("Run", "label run_dir game seed arm")


# ---- the reference and the engine ----------------------------------------------------------------
_ARC = None


def arcade():
    """The local engine, set up as run_local.py sets it up (offline). One per process."""
    global _ARC
    if _ARC is None:
        import run_local
        _ARC = run_local.make_arcade(offline=True)
    return _ARC


def game_seed(run_dir):
    """The seed run_local.py gave the engine for this run."""
    with open(os.path.join(run_dir, "run_config.json")) as f:
        return json.load(f)["seed"] % (2 ** 31 - 1)


def engine_move(index):
    """A recorded move (0-4 = ACTION1-5, 5 + 64*y + x = a click at x, y) as the engine's (action, data)."""
    from arcengine import GameAction
    if index < 5:
        return getattr(GameAction, f"ACTION{index + 1}"), {}
    y, x = divmod(int(index) - 5, GRID)
    return GameAction.ACTION6, {"x": x, "y": y}


def last_frame(obs):
    return np.array(obs.frame)[-1]


class Wrapped:
    """The arcade, handing the reference's run_episode a wrapped environment (and keeping hold of it)."""

    def __init__(self, arc, wrap):
        self.arc, self.wrap, self.env = arc, wrap, None

    def make(self, game_id, seed=0, scorecard_id=None, save_recording=False):
        self.env = self.wrap(self.arc.make(game_id, seed=seed))
        return self.env


class CountingEnv:
    """The environment, counting actions the way run_episode counts them (every
    step, and every reset after the first) and noting the count at each
    level-up. run_episode keeps the first three level-ups only, and is gone
    when a prefix ends early."""

    def __init__(self, env):
        self.env, self.actions, self.resets, self.levels, self.level_ups = env, 0, 0, 0, []

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        obs = self.env.reset()
        self.actions += self.resets > 0
        self.resets += 1
        return obs

    def step(self, action, data=None, reasoning=None):
        obs = self.env.step(action, data=data, reasoning=reasoning)
        self.actions += 1
        if obs is not None and obs.levels_completed > self.levels:
            self.levels = obs.levels_completed
            self.level_ups.append((self.levels, self.actions))
        return obs


class Differs(Exception):
    """The reference in its own harness did not make the run's move."""


class RecordingEnded(Exception):
    """Every recorded move has been made again. The argument says who asked for the next one."""


class FollowingEnv:
    """The environment, held to a recording: every move made on it must be the
    recording's next move, and every screen the recorded one. It notes which
    moves the reference chose itself (run_episode passes each move's origin on
    as `reasoning`)."""

    def __init__(self, env, rec):
        self.env, self.rec, self.i, self.obs, self.chosen = env, rec, 0, None, []

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self, action, data=None, reasoning=None):
        i, own = self.i, not reasoning["raw"].startswith("recorded")
        if i >= len(self.rec["action_nums"]):
            raise RecordingEnded("the reference" if own else "its explorer")
        num = int(self.rec["action_nums"][i])
        want, want_data = engine_move(int(self.rec["actions"][i]))
        if (action.name, data or {}) != (want.name, want_data):
            raise Differs(f"at action {num} the reference sent {action.name} {data or ''}, the run {want.name} {want_data or ''}")
        if not np.array_equal(last_frame(self.obs), self.rec["frames"][i]):
            raise Differs(f"before action {num} the engine's screen is not the recorded one")
        self.obs = self.env.step(action, data=data, reasoning=reasoning)
        if not np.array_equal(last_frame(self.obs), self.rec["next_frames"][i]):
            raise Differs(f"after action {num} the engine's screen is not the recorded one")
        self.chosen += [num] if own else []
        self.i += 1
        return self.obs


# ---- recordings ------------------------------------------------------------------------------------
def read_runs(specs, games, registered):
    """The recordings to use: RUNS, or --runs' "<manifest>[:<arm>]" and run folders. Games outside dev
    and seen are refused."""
    wanted = RUNS if not specs else [(os.path.basename(os.path.dirname(s.split(":")[0])) or s, s.split(":")[0],
                                      s.split(":")[1] if ":" in s else None) for s in specs]
    runs = []
    for label, path, arm in wanted:
        if os.path.isdir(path):                 # one run folder (a single CPU run has no manifest)
            with open(os.path.join(path, "run_config.json")) as f:
                runs.append(Run(os.path.basename(os.path.dirname(path)), path, json.load(f)["game_id"], "-", arm or "-"))
            continue
        for r in _manifest.read_manifest(path):
            if (arm is None or r["arm"] == arm) and (not games or r["game"] in games):
                runs.append(Run(label, r["run_dir"], r["game"], int(r["seed"]), r["arm"]))
    runs = [r for r in runs if not games or r.game in games]
    held_out = sorted({r.game for r in runs} - set(tiers.BUILD_FROM))
    if held_out and not registered:
        if games:
            raise SystemExit(f"refused: {', '.join(held_out)} are not dev or seen games (tiers.py). "
                             f"The other tiers are read only inside the test (--registered).")
        runs = [r for r in runs if r.game in tiers.BUILD_FROM]
    return runs


def read_recording(run_dir, upto):
    """A run's recorded moves in the order they were made, up to action number `upto` (None = all of them)."""
    names = ("frames", "actions", "next_frames", "levels", "action_nums", "game_over")
    parts = {k: [] for k in names}
    for path in sorted(glob.glob(os.path.join(run_dir, "transitions", "*.npz"))):
        with np.load(path) as z:
            for k in names:
                parts[k].append(z[k])
            if upto is not None and len(z["action_nums"]) and z["action_nums"][-1] >= upto:
                break
    rec = {k: np.concatenate(v) for k, v in parts.items()}
    keep = rec["action_nums"] <= (upto if upto is not None else rec["action_nums"].max(initial=0))
    return {k: v[keep] for k, v in rec.items()}


# ---- 1. repro: the report's table on this machine --------------------------------------------------
def repro_one(task):
    name, game_id, seed, out = task
    e18, _, e20 = load_reference()
    if name == "E0":
        agent = e18.StochasticGooseAgent(seed=seed)
    elif name == "E18":
        agent = e18.KnowledgePlannerAgent(seed=seed)
    else:
        agent = e20.LearnedKnowledgeAgent(seed=seed, knowledge_path=os.path.join(
            out, "repro_knowledge", f"{game_id}_s{seed}.json"))
    row = e18.run_episode(arcade(), game_id, agent, None, max_steps=e18.BUDGET, seed=seed)
    return dict(agent=name, game=game_id, seed=seed, levels=row["levels"], actions=row["actions"], state=row["state"],
                first_clear=row["first_clear"], second_clear=row["second_clear"], third_clear=row["third_clear"],
                clears=row.get("level_clears_log", ""), seconds=row["seconds"])


def repro(a):
    e18, _, _ = load_reference()
    seeds = [int(s) for s in a.seeds.split(",")]
    tasks = [(name, g, s, a.out) for name in ("E0", "E18", "E20") for g in e18.LIVE_GAMES for s in seeds]
    rows = sorted(run_all(repro_one, tasks, a.jobs, lambda r: f"{r['agent']} {r['game']} s{r['seed']}: "
                          f"levels {r['levels']} (first clear {r['first_clear']}), {r['seconds']:.0f} s"),
                  key=lambda r: (r["game"], r["agent"], r["seed"]))
    write_tsv(os.path.join(a.out, "repro.tsv"), rows)
    lines = [md(["Game", "Agent"] + [f"s{s}" for s in seeds] + ["Mean", "The report"]), md(["---"] * (len(seeds) + 4))]
    for g in e18.LIVE_GAMES:
        for name in ("E0", "E18", "E20"):
            got = [r["levels"] for r in rows if r["game"] == g and r["agent"] == name]
            want = [REPORT[g, name][s] for s in seeds]
            lines.append(md([g, name] + got + [f"{np.mean(got):.2f}", " ".join(str(v) for v in want)
                                               + (" (same)" if got == want else " (differs)")]))
    for name in ("E0", "E18", "E20"):
        got = [r["levels"] for r in rows if r["agent"] == name]
        want = [REPORT[g, name][s] for g in e18.LIVE_GAMES for s in seeds]
        lines.append(md(["both", name] + [""] * len(seeds) + [f"{np.mean(got):.2f}", f"mean {np.mean(want):.2f}"]))
    # the bar of Phase 0.2: E20's mean, and every episode past level 1 reaches level 6 on ft09 and 4 on lp85
    e20_rows = [r for r in rows if r["agent"] == "E20"]
    mean = np.mean([r["levels"] for r in e20_rows])
    short = [f"{r['game']} s{r['seed']} ({r['levels']})" for r in e20_rows
             if r["levels"] >= 1 and r["levels"] < (6 if r["game"].startswith("ft09") else 4)]
    lines += ["", f"E20's mean: {mean:.2f} levels per episode over {len(e20_rows)} episodes (the bar: at least 4.0; "
              f"the report: 4.33). Episodes that cleared level 1 and stopped short of level 6 (ft09) or 4 (lp85): "
              f"{', '.join(short) if short else 'none'}.",
              f"**Phase 0.2: {'PASS' if mean >= 4.0 and not short else 'FAIL'}**"]
    finish(os.path.join(a.out, "repro.md"), lines)


# ---- 2. replay-check: can a recording be re-driven, and what does a level-completing move return? ----
def replay_one(task):
    run, upto = task
    e18, _, e20 = load_reference()
    rec = read_recording(run.run_dir, upto)
    env = arcade().make(run.game, seed=game_seed(run.run_dir))
    obs, count, level, layout = env.reset(), 0, 0, None
    out = dict(run._asdict(), rows=len(rec["action_nums"]), matched=0, mismatch="", level_ups=[])
    for i, num in enumerate(rec["action_nums"].tolist()):
        while count + 1 < num:                  # a RESET is an action but not a row (after a game over)
            obs, count = env.reset(), count + 1
        cur = last_frame(obs)
        if layout is None:                      # the level's slots, as the reference takes them: its first screen
            layout = e18.LevelLayout(cur.astype(int), e20.MajoritySigTable())
        if not np.array_equal(cur, rec["frames"][i]) or int(rec["levels"][i]) != level:
            out["mismatch"] = f"before action {num}"
            break
        action, data = engine_move(int(rec["actions"][i]))
        obs, count = env.step(action, data=data), count + 1
        if not np.array_equal(last_frame(obs), rec["next_frames"][i]):
            out["mismatch"] = f"after action {num}"
            break
        out["matched"] += 1
        if obs.levels_completed > level:        # what the level-completing move returned, against the old level's slots
            stack = np.array(obs.frame)
            fits = [layout.encode(f.astype(int))[0] is not None for f in stack[:-1]]
            out["level_ups"].append(dict(level=level + 1, action=num, frames=len(stack), fits=fits))
            level, layout = obs.levels_completed, None
    return out


def replay_check(a):
    runs = read_runs(a.runs, split(a.games), a.registered)
    done = run_all(replay_one, [(r, None if a.all else a.actions) for r in runs], a.jobs,
                   lambda r: f"{r['label']} {r['arm']} {r['game']} s{r['seed']}: {r['matched']:,} of {r['rows']:,} moves"
                             f"{', MISMATCH ' + r['mismatch'] if r['mismatch'] else ''}, {len(r['level_ups'])} level-ups")
    done.sort(key=lambda r: (r["game"], r["label"], r["arm"], r["seed"]))
    write_tsv(os.path.join(a.out, "replay_check.tsv"),
              [{k: v for k, v in r.items() if k != "level_ups"} for r in done])
    ups = [dict(label=r["label"], arm=r["arm"], game=r["game"], seed=r["seed"], level=u["level"], action=u["action"],
                frames=u["frames"], second_last_fits=bool(u["fits"][-1]) if u["fits"] else "",
                earlier_frames_that_fit=sum(u["fits"])) for r in done for u in r["level_ups"]]
    write_tsv(os.path.join(a.out, "level_ups.tsv"), ups)
    lines = ["| Game | Recordings | Moves replayed | Moves matched | Recordings with a mismatch | Level-ups seen "
             "| Frames a level-completing move returned | ... with a frame before the last | ... whose second-to-last "
             "frame fits the old level's slots |", "|---|---|---|---|---|---|---|---|---|"]
    for g in sorted({r["game"] for r in done}):
        rs, us = [r for r in done if r["game"] == g], [u for u in ups if u["game"] == g]
        bad = [f"{r['label']} {r['arm']} s{r['seed']} ({r['mismatch']})" for r in rs if r["mismatch"]]
        frames = sorted({u["frames"] for u in us})
        lines.append(f"| {g} | {len(rs)} | {sum(r['rows'] for r in rs):,} | {sum(r['matched'] for r in rs):,} | "
                     f"{len(bad)}{': ' + '; '.join(bad[:3]) if bad else ''} | {len(us)} | "
                     f"{(str(frames[0]) if len(frames) == 1 else f'{frames[0]} to {frames[-1]}') if frames else '-'} | "
                     f"{sum(u['frames'] > 1 for u in us)} of {len(us)} | "
                     f"{sum(u['second_last_fits'] is True for u in us)} of {len(us)} |")
    finish(os.path.join(a.out, "replay_check.md"), lines)


# ---- 3. prefix: the reference's planner after a recorded Goose level 1 -----------------------------
class GaveUp(Exception):
    """The reference asked its explorer for a move after the recording had ended."""


class Mismatch(Exception):
    """The engine's screen is not the recorded one: the recording did not replay."""


def prefix_one(task):
    run, cap, out = task
    e18, _, e20 = load_reference()
    t0 = time.time()
    rec = read_recording(run.run_dir, cap)
    res = dict(run._asdict(), first_level_up="", goose_levels=int(rec["levels"].max(initial=0)), levels="",
               level_actions="", stopped="", solved_by="", top_rule="", seconds=0.0)
    later = np.nonzero(rec["levels"] >= 1)[0]
    if not len(later):
        res["stopped"] = f"no prefix: no level 1 inside {cap:,} actions"
        return res
    last = int(later[0]) - 1                    # the level-completing move: the last row recorded on level 1
    res["first_level_up"] = int(rec["action_nums"][last])
    feed = iter(range(last + 1))

    def recorded(self, obs, legal):
        """E0's act with the recording choosing: E0's own bookkeeping, then the recording's next move."""
        self._check_level_up(obs)
        self.step_count += 1
        self.level_actions += 1
        i = next(feed, None)
        if i is None:
            raise GaveUp(why_fell_back(e18, self, e18.get_grid(obs)) if self.level
                         else "the recorded level-up did not happen")
        if not np.array_equal(e18.get_grid(obs), rec["frames"][i]):
            raise Mismatch(int(rec["action_nums"][i]))
        action, data = engine_move(int(rec["actions"][i]))
        return next(x for x in legal if x.name == action.name), data, f"recorded:{action.name}", True

    tag = f"{run.label}_{run.arm}_{run.game}_s{run.seed}"
    agent = e20.LearnedKnowledgeAgent(seed=0, knowledge_path=os.path.join(out, "prefix", tag, "store.json"))
    arc, e0_act = Wrapped(arcade(), CountingEnv), e18.StochasticGooseAgent.act
    e18.StochasticGooseAgent.act = recorded
    try:        # the cap is in the recording's action numbers: its first move is numbered 1 (2 after a starting RESET)
        row = e18.run_episode(arc, run.game, agent, None, seed=game_seed(run.run_dir),
                              max_steps=cap - (int(rec["action_nums"][0]) - 1))
        res["stopped"] = "won the game" if row["state"] == "WIN" else f"the cap, on level {row['levels'] + 1}"
    except GaveUp as e:
        res["stopped"] = f"gave up on level {agent.level + 1}: {e}"
    except Mismatch as e:
        res["stopped"] = f"REPLAY MISMATCH before action {e}"
    finally:
        e18.StochasticGooseAgent.act = e0_act
    ups = [(0, 0)] + arc.env.level_ups
    res.update(levels=arc.env.levels, seconds=round(time.time() - t0, 1),
               level_actions=" ".join(str(b[1] - a[1]) for a, b in zip(ups[1:], ups[2:])),
               solved_by=" | ".join(f"L{h['level']}: {h['solved_by']}" for h in agent.store.history))
    ranked = agent.store.ranked()
    if ranked:
        res["top_rule"] = (f"{agent.store.hyps[ranked[0]].describe()} (confirmed on "
                           f"{agent.store.stats[ranked[0]]['confirmed']} levels)")
    os.makedirs(os.path.join(out, "prefix", tag), exist_ok=True)
    with open(os.path.join(out, "prefix", tag, "log.json"), "w") as f:      # the reference's own two logs
        json.dump(dict(result=res, log=[str(x) for x in agent.log],
                       knowledge_log=[str(x) for x in agent.knowledge_log]), f, indent=1, default=str)
    return res


def whole_one(task):
    """Phase 1.4 and 1.6. The recording of a run made with EVAL_KP=1 stands in for the reference's
    explorer for the whole run. Both sides are deterministic, so the reference in its own harness must
    choose exactly the moves the bridge reported as planned (kp_stats.json, planned_at) with the
    clicks the run sent, and ask its explorer for every other move."""
    run, out = task
    e18, _, e20 = load_reference()
    rec = read_recording(run.run_dir, None)
    with open(os.path.join(run.run_dir, "kp", "kp_stats.json")) as f:
        stats = json.load(f)
    with open(os.path.join(run.run_dir, "run_end.json")) as f:
        made = json.load(f)["n_actions"]                # every action of the run, RESETs included
    planned = [c + 1 for c in stats["planned_at"]]      # Goose's counter before a move is the move's number less one
    res = dict(run._asdict(), actions=made, rows=len(rec["action_nums"]), replayed=0, planned_in_run=len(planned),
               planned_in_replay=0, differs="", levels=0, kp_errors=stats["kp_errors"])

    def recorded(self, obs, legal):
        self._check_level_up(obs)
        self.step_count += 1
        self.level_actions += 1
        i = arc.env.i
        if i >= len(rec["action_nums"]):
            raise RecordingEnded("its explorer")
        action, data = engine_move(int(rec["actions"][i]))
        return next(x for x in legal if x.name == action.name), data, f"recorded:{action.name}", True

    arc, e0_act = Wrapped(arcade(), lambda env: FollowingEnv(env, rec)), e18.StochasticGooseAgent.act
    agent = e20.LearnedKnowledgeAgent(seed=0, knowledge_path=os.path.join(out, "whole", f"{run.label}_{run.game}", "store.json"))
    e18.StochasticGooseAgent.act = recorded
    asked = None
    try:
        e18.run_episode(arc, run.game, agent, None, seed=game_seed(run.run_dir), max_steps=made + 10)
    except RecordingEnded as e:
        asked = str(e)
    except Differs as e:
        res["differs"] = str(e)
    finally:
        e18.StochasticGooseAgent.act = e0_act
    chosen, last = list(arc.env.chosen), int(rec["action_nums"][-1])
    # The run's last move is never a row (nothing saw its result). If it was a move and not a RESET, the
    # reference must want the next move from whoever the bridge says made it.
    if not res["differs"] and made > last + int(rec["game_over"][-1]):
        if asked is None:
            res["differs"] = "the replay ended before the run's last move"
        elif (asked == "the reference") != (made in planned):
            res["differs"] = f"the run's last move (action {made}) was asked of {asked}"
        chosen += [made] if asked == "the reference" else []
    if not res["differs"] and chosen != planned:
        first = next((a for a, b in zip(chosen + [None], planned + [None]) if a != b), None)
        res["differs"] = f"the planned moves part at action {first}"
    res.update(replayed=arc.env.i, planned_in_replay=len(chosen), levels=int(agent.level))
    return res


def whole(a):
    runs = read_runs(a.runs, split(a.games), a.registered)
    done = run_all(whole_one, [(r, a.out) for r in runs], a.jobs,
                   lambda r: f"{r['label']} {r['game']}: {r['replayed']:,} of {r['rows']:,} moves, "
                             f"{r['planned_in_replay']:,} planned (the run: {r['planned_in_run']:,}) "
                             f"{'DIFFERS: ' + r['differs'] if r['differs'] else 'the same'}")
    done.sort(key=lambda r: (r["game"], r["label"]))
    write_tsv(os.path.join(a.out, "whole.tsv"), done)
    lines = ["| Run | Game | Actions | Moves recorded | Moves replayed | Planned clicks: the run | ... the reference "
             "in its own harness | Errors in the run | Result |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [md([r["label"], r["game"], f"{r['actions']:,}", f"{r['rows']:,}", f"{r['replayed']:,}",
                  f"{r['planned_in_run']:,}", f"{r['planned_in_replay']:,}", r["kp_errors"],
                  f"DIFFERS: {r['differs']}" if r["differs"] else "the same moves, to the last one"]) for r in done]
    same = all(not r["differs"] and r["replayed"] == r["rows"] for r in done)
    lines += ["", f"**The bridge against the reference: {'PASS' if done and same else 'FAIL'}**"]
    finish(os.path.join(a.out, "whole.md"), lines)


def prefix(a):
    if a.whole:
        return whole(a)
    games = split(a.games) or ["ft09", "lp85"]
    runs = read_runs(a.runs, games, a.registered)
    done = run_all(prefix_one, [(r, a.actions, a.out) for r in runs], a.jobs,
                   lambda r: f"{r['label']} {r['arm']} {r['game']} s{r['seed']}: level 1 at {r['first_level_up']}, "
                             f"{r['levels']} levels, {r['stopped']} ({r['seconds']:.0f} s)")
    order = {arm: i for i, arm in enumerate(OBJCLICK)}      # SOT's row first, then the other objclick arms, then the rest
    done.sort(key=lambda r: (r["game"], order.get(r["arm"], 9), r["label"], r["arm"], r["seed"]))
    write_tsv(os.path.join(a.out, "prefix.tsv"), done)
    lines = []
    for g in games:
        rows = [r for r in done if r["game"] == g]
        lines += [f"### {g}", "", "| Recording | Arm | Seed | Level 1 at action | Levels by action "
                  f"{a.actions:,}: the planner | ... the recording's Goose | Actions for each planned level "
                  "| How it stopped |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['label']} | {r['arm']} | {r['seed']} | {r['first_level_up']} | {r['levels']} | "
                  f"{r['goose_levels']} | {r['level_actions']} | {r['stopped']} |" for r in rows]
        lines.append("")
    lines += verdict(done, games) if not a.runs else ["(The go/no-go is read on the plan's full list: run without --runs.)"]
    finish(os.path.join(a.out, "prefix.md"), lines)


def verdict(done, games):
    """The go/no-go of Phase 0.4, by the bars fixed in the plan (GO)."""
    lines, ok = [], True
    for g in [g for g in games if g in GO]:
        need, share, of_objclick = GO[g]
        rows = [r for r in done if r["game"] == g and r["first_level_up"] != "" and "MISMATCH" not in r["stopped"]]
        left_out = [r for r in done if r["game"] == g and r not in rows]
        hit = [r for r in rows if r["levels"] >= need]
        oc = [r for r in rows if r["arm"] in OBJCLICK]
        oc_hit = [r for r in oc if r["levels"] >= need]
        passed = len(rows) > 0 and len(hit) >= share * len(rows) and len(oc_hit) >= of_objclick
        ok = ok and passed
        lines.append(f"- **{g}: {'PASS' if passed else 'FAIL'}.** Level {need} or more from {len(hit)} of {len(rows)} "
                     f"prefixes ({len(hit) / max(1, len(rows)):.0%}; the bar: {share:.0%}), and from {len(oc_hit)} of "
                     f"the {len(oc)} played with `objclick` (the bar: {of_objclick}). "
                     f"Not counted: {len(left_out)}"
                     f"{' (' + '; '.join(sorted({r['stopped'] for r in left_out})) + ')' if left_out else ''}.")
    return lines + ["", f"**Phase 0.4, the go/no-go: {'PASS' if ok else 'FAIL'}**"]


# ---- shared ------------------------------------------------------------------------------------------
def split(text):
    return [x for x in (text or "").split(",") if x]


def md(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def start_worker():
    os.chdir(ROOT)                              # the engine finds environment_files/ from here
    logging.disable(logging.CRITICAL)           # the engine logs every game it loads
    import torch
    torch.set_num_threads(1)


def run_all(fn, tasks, jobs, say):
    """Each task in a fresh worker process (PYTHONHASHSEED=0, one thread, no GPU), `jobs` at a time."""
    os.environ.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    done, t0 = [], time.time()
    with mp.get_context("spawn").Pool(min(jobs, max(1, len(tasks))), initializer=start_worker,
                                      maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(fn, tasks):
            done.append(r)
            print(f"  [{len(done):>3}/{len(tasks)}  {time.time() - t0:5.0f} s] {say(r)}", flush=True)
    return done


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(rows[0]) if rows else []
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


def finish(path, lines):
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    print("\n" + text + f"\n\nwrote {path} (and the .tsv beside it)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("repro", "replay-check", "prefix"):
        p = sub.add_parser(name)
        p.add_argument("--jobs", type=int, default=8, help="episodes at a time (default 8)")
        p.add_argument("--out", default=OUT)
        if name == "repro":
            p.add_argument("--seeds", default="0,1,2,3,4,5", help="the report's six")
            continue
        p.add_argument("--runs", nargs="*", default=[], help="<manifest>[:<arm>] ... (default: the plan's list)")
        p.add_argument("--games", default="",
                       help="comma-separated (default: ft09,lp85 for prefix, every dev and seen game otherwise)")
        p.add_argument("--actions", type=int, default=CAP, help=f"the budget, in the recording's action numbers ({CAP:,})")
        p.add_argument("--registered", action="store_true", help="allow games outside dev and seen (the test only)")
        if name == "replay-check":
            p.add_argument("--all", action="store_true", help="replay whole recordings, not their first --actions")
        if name == "prefix":
            p.add_argument("--whole", action="store_true",
                           help="the bridge against the reference: --runs are runs made with EVAL_KP=1")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(a.out, exist_ok=True)
    {"repro": repro, "replay-check": replay_check, "prefix": prefix}[a.cmd](a)


if __name__ == "__main__":
    main()
