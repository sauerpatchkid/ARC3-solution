#!/usr/bin/env python3
"""run_curriculum.py — play several games back-to-back with ONE persistent brain.

This is the cross-game generalization harness. Unlike run_local.py (one game,
one fresh agent) or the sweep scripts (each game a separate process from
scratch), this keeps a SINGLE StochasticGoose brain alive across the whole
game list:

    game 1  →  game 2  →  game 3  →  ...
    └──────── same model / optimizer / experience buffer ────────┘

So whatever the agent learns on game 1 carries into game 2, and so on. If the
algorithm genuinely generalizes, later games should be solved with FEWER
actions than they would from a cold start — watch `first_levelup_action` fall
down the sequence in the summary table.

------------------------------------------------------------------------------
Quick start
------------------------------------------------------------------------------
Edit the GAMES list below, or pass --games. Same env contract as everything
else in the repo (EVAL_SEED / EVAL_MAX_ACTIONS / PYTHONHASHSEED):

    EVAL_SEED=0 EVAL_MAX_ACTIONS=200000 PYTHONHASHSEED=0 \
        uv run python run_curriculum.py --games ft09,dc22,ls20

Outputs, all under one runs/<timestamp>/ tree:
    <game>/transitions/   the .npz corpus for that leg of the curriculum
    <game>/metrics.json   per-game scores (from compute_metrics.compute)
    <game>/tensorboard/   per-game learning curves
    curriculum_summary.md / .csv   the transfer table (read this)

Optional flags: --suite curriculum_suite.csv (append per-game metric rows),
--offline, --render terminal.

------------------------------------------------------------------------------
How to read the result (be honest in the write-up)
------------------------------------------------------------------------------
A falling `first_levelup_action` across the sequence is SUGGESTIVE of transfer,
but it is confounded by game ordering and difficulty. The clean control is the
same games played cold (which the solo sweeps already produce): transfer =
curriculum result minus cold-start result. This script gives you the curriculum
half; pair it with a from-scratch run of the same games to make the claim.

Design note: this file imports and reuses run_local.py's engine glue and
compute_metrics.py's scorer, and never edits either — so it can be developed
while a sweep is running without perturbing the baseline.
"""
import argparse
import os
import time

import run_local  # import side effects: sys.path setup, arc_agi, minimal agents pkg
from eval_common import (resolve_seed, resolve_max_actions, write_run_config,
                         TransitionLogger)
from compute_metrics import compute, append_suite
from utils import get_environment_directory
from torch.utils.tensorboard import SummaryWriter

# Default curriculum if --games is omitted: learnable → mixed → null contrast.
# ft09 reliably completes levels; dc22 is a mixed action space; ls20 completes
# nothing (pure exploration). Edit freely — order is the experiment.
GAMES = ["ft09", "dc22", "ls20"]


def _unique_env_dir(base_dir, game):
    """A per-game output dir under the run tree. If the same game appears twice
    in the curriculum, suffix _2, _3, ... so legs never clobber each other."""
    name = game
    k = 2
    while os.path.exists(os.path.join(base_dir, name)):
        name = f"{game}_{k}"
        k += 1
    return get_environment_directory(base_dir, name)


def _write_curriculum_config(env_dir, agent, game, position, games, cap):
    """Record exactly how this leg was configured (mirrors action.py's
    run_config, plus the curriculum position + full game list)."""
    write_run_config(
        env_dir,
        agent="stochastic_goose",
        game_id=game,
        seed=agent.seed,
        seed_source=agent.seed_source,
        max_actions=cap,
        reset_on_level=agent.reset_on_level,
        train_frequency=agent.train_frequency,
        batch_size=agent.batch_size,
        buffer_capacity=agent.experience_buffer.maxlen,
        persist_brain=True,
        curriculum_position=position,
        curriculum_games=list(games),
    )


def begin_game(agent, game, position, games, cap):
    """Re-point the persistent agent at a new game WITHOUT touching its brain.

    Kept across the boundary: action_model, optimizer, experience_buffer,
    experience_hashes (the whole point). Reset: per-game counters and the
    previous-frame trackers, so no transition is logged spanning two games.
    """
    env_dir = _unique_env_dir(agent.base_dir, game)
    agent.game_id = game
    agent.log_dir = env_dir
    # Fresh corpus + TensorBoard writer for this leg.
    agent.transition_logger = TransitionLogger(os.path.join(env_dir, "transitions"))
    agent.writer = SummaryWriter(os.path.join(env_dir, "tensorboard"))
    # Per-game bookkeeping reset (brain untouched).
    agent.action_counter = 0
    agent.current_score = -1
    agent.prev_frame = None
    agent.prev_action_idx = None
    agent.prev_frame_raw = None
    agent._last_decision_time = None
    _write_curriculum_config(env_dir, agent, game, position, games, cap)
    return env_dir


def play_game(agent, env, cap):
    """Run one leg of the curriculum to the action cap (or WIN / 8h).

    This mirrors run_local.main()'s inner loop on purpose — run_local is left
    untouched so the live sweep that imports it is unaffected."""
    obs = env.reset()
    frame = run_local.ShimFrame(obs, getattr(env, "action_space", None))
    t0 = time.time()
    consecutive_resets = 0
    while agent.action_counter < cap:
        if agent.is_done([frame], frame):
            print(f"  [curriculum] is_done at {agent.action_counter} "
                  f"(state={frame.state.name})")
            break

        action = agent.choose_action([frame], frame)
        eng_action, data, is_reset = run_local.to_engine_action(
            action, getattr(agent, "prev_action_idx", None))

        if is_reset:
            obs = env.reset()
            consecutive_resets += 1
            if consecutive_resets > 10:
                raise SystemExit("10+ resets with no progress — check reset "
                                 "semantics / initial state handling")
        else:
            obs = env.step(eng_action, data=data)
            consecutive_resets = 0

        frame = run_local.ShimFrame(obs, getattr(env, "action_space", None))
        agent.action_counter += 1

        if agent.action_counter % 1000 == 0:
            aps = agent.action_counter / (time.time() - t0)
            print(f"    {agent.action_counter:>7} actions  score={frame.score}  "
                  f"{aps:5.1f} act/s")


def score_leg(env_dir, game, agent, suite, seed_label):
    """Score a finished leg in-process with compute_metrics.compute. Returns the
    metrics dict, or None if the leg produced no usable corpus."""
    corpus = os.path.join(env_dir, "transitions")
    try:
        m = compute(corpus)
    except SystemExit as e:            # compute raises SystemExit on empty corpus
        print(f"  [curriculum] no metrics for {game}: {e}")
        return None
    except Exception as e:
        print(f"  [curriculum] scoring failed for {game}: {e}")
        return None
    import json
    with open(os.path.join(env_dir, "metrics.json"), "w") as f:
        json.dump({"game": game, "agent": "goose_curriculum",
                   "seed": seed_label, **m}, f, indent=2)
    if suite:
        append_suite(suite, m, game, "goose_curriculum", seed_label)
    lvl = m["levels_completed"]
    first = m["first_levelup_action"]
    print(f"  [curriculum] {game}: levels={lvl} first_levelup={first} "
          f"meaningful_change={m['meaningful_change_rate']} "
          f"unique_states={m['unique_states']}")
    return m


def write_summary(base_dir, results, games, seed_label):
    """Write the transfer table (curriculum_summary.md / .csv) and print it."""
    cols = ["position", "game", "n_actions", "levels_completed",
            "first_levelup_action", "meaningful_change_rate", "unique_states",
            "redundancy", "actions_per_sec"]

    def cell(m, k):
        if m is None:
            return "—"
        v = m.get(k)
        return "—" if v is None else v

    lines = ["# Curriculum summary (persistent brain)", ""]
    lines.append(f"- games (in order): {', '.join(games)}")
    lines.append(f"- base seed label: {seed_label}")
    lines.append("")
    header = "| # | game | actions | levels | first_levelup | meaningful_chg | unique | redundancy | act/s |"
    sep = "|---|------|--------:|-------:|--------------:|---------------:|-------:|-----------:|------:|"
    lines += [header, sep]
    for pos, game, m in results:
        lines.append(
            f"| {pos} | {game} | {cell(m,'n_actions')} | {cell(m,'levels_completed')} "
            f"| {cell(m,'first_levelup_action')} | {cell(m,'meaningful_change_rate')} "
            f"| {cell(m,'unique_states')} | {cell(m,'redundancy')} "
            f"| {cell(m,'actions_per_sec')} |")
    lines += [
        "",
        "**Reading this:** a falling `first_levelup` down the sequence suggests "
        "the brain is transferring, but it is confounded by game order/difficulty. "
        "The clean control is the same games played cold — transfer = curriculum "
        "minus cold-start (see the solo sweeps).",
    ]

    md_path = os.path.join(base_dir, "curriculum_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    import csv
    csv_path = os.path.join(base_dir, "curriculum_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pos, game, m in results:
            w.writerow([pos, game] + [
                ("" if (m is None or m.get(k) is None) else m.get(k)) for k in cols[2:]])

    print("\n" + "\n".join(lines))
    print(f"\n[curriculum] wrote {md_path} and {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Persistent-brain multi-game runner.")
    ap.add_argument("--games", default=None,
                    help="comma-separated game ids in play order "
                         f"(default: {','.join(GAMES)})")
    ap.add_argument("--suite", default=None,
                    help="optional CSV to append per-game metric rows to "
                         "(agent label: goose_curriculum)")
    ap.add_argument("--offline", action="store_true",
                    help="OperationMode.OFFLINE (needs locally cached games)")
    ap.add_argument("--render", default=None, choices=[None, "terminal", "human"])
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",")] if args.games else list(GAMES)
    if not games:
        raise SystemExit("no games to play (empty --games)")
    cap = resolve_max_actions()
    seed_label = os.getenv("EVAL_SEED", "NA")

    # Build the ONE brain (seeded once, from the first game's id). Everything
    # after this reuses it — no second Action is ever constructed.
    agent = run_local.build_agent(games[0])
    # Persistent-brain mode REQUIRES the persist arm: otherwise the first
    # level-up inside a game would wipe the transferred brain. This selects the
    # existing EVAL_RESET_ON_LEVEL=0 behavior; it does not change what the flag
    # means for run_local / the API path.
    if agent.reset_on_level:
        print("[curriculum] forcing reset_on_level=False for persistent-brain mode "
              "(model/optimizer/buffer carry across every level AND game)")
    agent.reset_on_level = False

    print(f"[curriculum] games={games} cap={cap} seed_label={seed_label} "
          f"device={getattr(agent, 'device', '?')}")
    print(f"[curriculum] run tree: {agent.base_dir}")

    results = []
    t0 = time.time()

    for i, game in enumerate(games, start=1):
        print(f"\n=== [{i}/{len(games)}] {game} "
              f"{'(persistent brain carried in)' if i > 1 else '(cold start)'} ===")
        if i == 1:
            # Reuse the dir/logger/writer Action.__init__ already made for game 1;
            # just stamp the curriculum metadata onto its run_config.
            env_dir = agent.log_dir
            _write_curriculum_config(env_dir, agent, game, i, games, cap)
        else:
            env_dir = begin_game(agent, game, i, games, cap)

        seed, _ = resolve_seed(game)
        game_seed = seed % (2 ** 31 - 1)
        # Fresh engine per game (mirrors the per-process sweep pattern): only the
        # BRAIN persists across games, the environment is always clean.
        arc = run_local.make_arcade(args.offline)
        env = run_local.make_env(arc, game, game_seed, args.render)
        if env is None:
            print(f"  [curriculum] failed to make env for {game} — skipping")
            results.append((i, game, None))
            continue

        play_game(agent, env, cap)
        agent.transition_logger.flush()
        try:
            agent.writer.flush()
        except Exception:
            pass
        m = score_leg(env_dir, game, agent, args.suite, seed_label)
        results.append((i, game, m))

    dt = time.time() - t0
    print(f"\n[curriculum] all {len(games)} games done in {dt:.1f}s")
    write_summary(agent.base_dir, results, games, seed_label)
    if args.suite:
        print(f"[curriculum] appended per-game rows to {args.suite}")


if __name__ == "__main__":
    main()
