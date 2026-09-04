#!/usr/bin/env python3
"""random_agent.py — the matched random-policy floor for the Team B comparison.

Why this exists: effective-action rate, redundancy, and state coverage are all
game-dependent, so the raw numbers are not comparable across the 25 games. A
random agent run on the SAME game and seed turns each of them into an uplift
ratio, which is what makes aggregating across heterogeneous games legitimate.
It is also the floor ARC itself calibrates against — an environment is only
accepted if a random policy solves a level less than about 1 in 10,000 attempts.

"Matched" is the important word: this samples uniformly over exactly the action
space StochasticGoose samples from — the masked 5 + 64x64 combined space, where
ACTION6 contributes all 4096 click coordinates as individual choices. It is not
uniform over {ACTION1..ACTION6}; that would be a different (and much stronger)
prior over clicking, and would not be a fair floor for a click-heavy agent.

It shares eval_common with every other agent, so it obeys the same seed, cap,
and corpus contract and produces a corpus compute_metrics.py scores identically.
No model, no training — model_ms is logged as 0.0 by design.

Run it exactly like the goose agent:
    EVAL_SEED=0 EVAL_MAX_ACTIONS=200000 PYTHONHASHSEED=0 \\
        uv run python run_local.py --game ft09 --agent random
"""
import atexit
import os
import random
import time

import numpy as np

from agents.structs import GameAction, GameState
from eval_common import (TransitionLogger, env_flag, resolve_max_actions,
                         resolve_seed, write_run_config)
from utils import get_environment_directory, setup_experiment_directory

GRID = 64


class RandomAgent:
    """Uniform-random policy over the masked 5 + 64x64 combined action space.

    Exposes the same surface run_local.py drives on the goose agent:
    game_id, action_counter, prev_action_idx, log_dir, transition_logger,
    is_done(), choose_action().
    """

    def __init__(self, game_id=None, **kwargs):
        self.game_id = game_id
        self.action_counter = 0
        self.seed, self.seed_source = resolve_seed(self.game_id)
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed % (2 ** 32 - 1))
        self.start_time = time.time()
        self.MAX_ACTIONS = resolve_max_actions()

        self.base_dir, _ = setup_experiment_directory()
        env_dir = get_environment_directory(self.base_dir, self.game_id)
        self.log_dir = env_dir

        self.log_transitions = env_flag("EVAL_LOG_TRANSITIONS", True)
        self.transition_logger = None
        if self.log_transitions:
            self.transition_logger = TransitionLogger(
                os.path.join(env_dir, "transitions"))
            atexit.register(self.transition_logger.flush)

        # Per-level reset is meaningless for a memoryless policy, but the flag is
        # recorded so the run_config matches the goose runs it is compared to.
        self.reset_on_level = env_flag("EVAL_RESET_ON_LEVEL", True)
        self.current_score = -1
        self.prev_frame_raw = None
        self.prev_action_idx = None
        self._last_decision_time = None

        write_run_config(
            env_dir,
            agent="random",
            game_id=self.game_id,
            seed=self.seed,
            seed_source=self.seed_source,
            max_actions=self.MAX_ACTIONS,
            log_transitions=self.log_transitions,
            reset_on_level=self.reset_on_level,
            policy="uniform over masked 5 + 64x64 combined action space",
        )
        print(f"Random agent logging to: {env_dir}")

    # --- the surface run_local.py uses ------------------------------------
    def _has_time_elapsed(self):
        return time.time() - self.start_time >= 8 * 3600 - 5 * 60

    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN or self._has_time_elapsed()

    def _valid_indices(self, available_actions):
        """Unified indices for the available actions: 0-4 = ACTION1-5,
        5 + (64*y + x) = ACTION6 click at (x, y)."""
        names = {a.name for a in (available_actions or [])}
        idxs = [i for i in range(5) if f"ACTION{i + 1}" in names]
        if "ACTION6" in names:
            idxs.extend(range(5, 5 + GRID * GRID))
        if not idxs:                      # nothing advertised: fall back to the buttons
            idxs = list(range(5))
        return idxs

    def choose_action(self, frames, latest_frame):
        now = time.time()
        wall_ms = ((now - self._last_decision_time) * 1000.0
                   if self._last_decision_time else 0.0)
        self._last_decision_time = now

        current_frame_raw = np.array(latest_frame.frame, dtype=np.uint8)[-1]

        # Level boundary: flush and drop the previous-frame tracker so no logged
        # transition ever spans two levels (mirrors action.py).
        if latest_frame.score != self.current_score:
            if self.transition_logger is not None:
                self.transition_logger.flush()
            print(f"Score changed from {self.current_score} to "
                  f"{latest_frame.score} at action {self.action_counter}")
            self.prev_frame_raw = None
            self.prev_action_idx = None
            self.current_score = latest_frame.score

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_frame_raw = None
            self.prev_action_idx = None
            action = GameAction.RESET
            action.reasoning = "Game needs reset."
            return action

        # Log the transition the PREVIOUS action produced — every transition,
        # before any filtering, exactly as the contract requires.
        if self.prev_frame_raw is not None and self.transition_logger is not None:
            self.transition_logger.log(
                frame=self.prev_frame_raw,
                action_idx=self.prev_action_idx,
                next_frame=current_frame_raw,
                changed=not np.array_equal(self.prev_frame_raw, current_frame_raw),
                level=self.current_score,
                action_num=self.action_counter,
                wall_ms=wall_ms,
                model_ms=0.0,          # no model: the whole point of the floor
            )

        idx = self.rng.choice(self._valid_indices(latest_frame.available_actions))
        self.prev_frame_raw = current_frame_raw
        self.prev_action_idx = idx

        action = (GameAction.ACTION6 if idx >= 5
                  else getattr(GameAction, f"ACTION{idx + 1}"))
        action.reasoning = f"uniform random (index {idx})"
        return action
