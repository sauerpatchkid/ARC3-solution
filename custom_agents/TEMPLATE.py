"""TEMPLATE.py — copy this to start a new agent.

    cp custom_agents/TEMPLATE.py custom_agents/my_agent.py

then add one line to REGISTRY in custom_agents/__init__.py:

    "mine": ("my_agent", "MyAgent"),

and run it:

    EVAL_SEED=0 EVAL_MAX_ACTIONS=2000 PYTHONHASHSEED=0 \\
        uv run python run_local.py --game ft09 --agent mine

This file is a WORKING agent as written (it picks uniformly at random from the
buttons). Run it first, confirm you get a corpus and a metrics row, then start
replacing choose_action() with your own logic.

------------------------------------------------------------------------------
THE SURFACE run_local.py DRIVES

Your class must expose exactly these. Nothing else is required, and nothing
else is called.

    game_id             str, set from the constructor kwarg
    action_counter      int, INCREMENTED BY THE RUNNER — you only read it
    prev_action_idx     int or None: the unified index of the action you just
                        chose. The runner reads this to build the engine call,
                        so it must be set every time choose_action returns a
                        non-RESET action.
    log_dir             str, where your output goes (from utils below)
    transition_logger   TransitionLogger or None
    is_done(frames, latest_frame)    -> bool
    choose_action(frames, latest_frame) -> GameAction

`latest_frame` has four attributes: .frame (the 64x64 grid, possibly a stack of
animation frames — take [-1]), .score (levels completed so far), .state, and
.available_actions.

------------------------------------------------------------------------------
THE UNIFIED ACTION INDEX

One integer names every action, and it is the same integer everywhere — your
agent, the corpus, the metrics:

    0-4              ACTION1 .. ACTION5
    5 + (64*y + x)   ACTION6, a click at column x, row y

------------------------------------------------------------------------------
THE CONTRACT YOU MUST KEEP

These are what make your numbers comparable to everyone else's. Do not
reimplement them — import them, as below.

  * Seed from resolve_seed(game_id) and seed EVERY rng you use.
  * Stop at resolve_max_actions().
  * Log EVERY transition, before any filtering of your own.
  * Call write_run_config() so the run is reproducible from its own output.

See README.md and eval_common.py for the full contract.
"""
import atexit
import random
import time

import numpy as np

from agents.structs import GameAction, GameState
from eval_common import (TransitionLogger, env_flag, resolve_max_actions,
                         resolve_seed, write_run_config)
from utils import get_environment_directory, setup_experiment_directory

GRID = 64


class MyAgent:
    """Replace this docstring and choose_action() with your approach."""

    def __init__(self, game_id=None, **kwargs):
        self.game_id = game_id
        self.action_counter = 0          # the RUNNER increments this
        self.prev_action_idx = None

        # --- the eval contract: seed, cap, output directory ------------------
        self.seed, self.seed_source = resolve_seed(self.game_id)
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed % (2 ** 32 - 1))
        self.MAX_ACTIONS = resolve_max_actions()
        self.start_time = time.time()

        base_dir, _ = setup_experiment_directory()
        self.log_dir = get_environment_directory(base_dir, self.game_id)

        # --- the transition corpus: every metric is computed from this -------
        self.transition_logger = None
        if env_flag("EVAL_LOG_TRANSITIONS", True):
            self.transition_logger = TransitionLogger(
                f"{self.log_dir}/transitions")
            atexit.register(self.transition_logger.flush)

        # --- record exactly how this run was configured ----------------------
        # Add your own hyperparameters as extra keyword arguments; they land in
        # run_config.json, which is what makes an ablation reproducible.
        write_run_config(
            self.log_dir,
            agent="my_agent",
            game_id=self.game_id,
            seed=self.seed,
            seed_source=self.seed_source,
            max_actions=self.MAX_ACTIONS,
        )

        # State needed to log a transition once its outcome is known.
        self.prev_frame_raw = None
        self._last_decision_time = None
        self._last_model_ms = 0.0

    def is_done(self, frames, latest_frame):
        """Stop on a win, or after ~8 hours of wall clock."""
        return (latest_frame.state is GameState.WIN
                or time.time() - self.start_time >= 8 * 3600 - 300)

    def _valid_indices(self, available_actions):
        """Unified indices that are legal right now."""
        idx = []
        for a in available_actions or []:
            if 1 <= a.value <= 5:
                idx.append(a.value - 1)
            elif a.value == 6:
                idx.extend(range(5, 5 + GRID * GRID))
        return idx or [0]

    def choose_action(self, frames, latest_frame):
        now = time.time()
        wall_ms = ((now - self._last_decision_time) * 1000.0
                   if self._last_decision_time else 0.0)
        self._last_decision_time = now

        frame_raw = np.array(latest_frame.frame, dtype=np.uint8)[-1]

        # --- log the transition our PREVIOUS action produced -----------------
        if self.prev_frame_raw is not None and self.transition_logger is not None:
            self.transition_logger.log(
                frame=self.prev_frame_raw,
                action_idx=self.prev_action_idx,
                next_frame=frame_raw,
                changed=not np.array_equal(self.prev_frame_raw, frame_raw),
                level=latest_frame.score,
                action_num=self.action_counter,
                wall_ms=wall_ms,
                model_ms=self._last_model_ms,
            )

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.prev_frame_raw = None      # do not log across a reset
            return GameAction.RESET

        # ================= YOUR LOGIC GOES HERE ==============================
        t0 = time.time()
        idx = self.rng.choice(self._valid_indices(latest_frame.available_actions))
        self._last_model_ms = (time.time() - t0) * 1000.0
        # =====================================================================

        self.prev_frame_raw = frame_raw
        self.prev_action_idx = idx          # the runner reads this

        if idx < 5:
            return GameAction(idx + 1)
        coord = idx - 5
        y, x = divmod(coord, GRID)
        action = GameAction.ACTION6
        action.set_data({"x": int(x), "y": int(y)})
        return action
