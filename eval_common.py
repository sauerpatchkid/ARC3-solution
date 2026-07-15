"""eval_common.py — shared evaluation contract for ALL baseline agents.

Every agent in the Team B comparison (StochasticGoose, Blind Squirrel, random)
imports from this module. DO NOT copy/paste these into your agent — import them,
so the contract cannot drift between codebases:

    import sys, os
    sys.path.append(os.path.dirname(...))  # repo root
    from eval_common import (env_flag, stable_game_offset, resolve_seed,
                             resolve_max_actions, TransitionLogger,
                             write_run_config)

The contract (see README §5):
  EVAL_SEED            base seed; each game adds stable_game_offset(game_id)
  EVAL_MAX_ACTIONS     hard per-game action cap (0/unset = unlimited)
  EVAL_LOG_METRICS     TensorBoard scalars on/off        (default: on)
  EVAL_LOG_TRANSITIONS transition corpus on/off          (default: on)
  EVAL_SAVE_VIS        expensive debug visualizations    (default: off)

Schema authority: a corpus is valid iff `inspect_corpus.py` loads it without
errors. If your shards fail that script, your agent is out of contract.
"""
import hashlib
import json
import os

import numpy as np


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean flag from an environment variable ('1'/'true'/'yes'/'on')."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def stable_game_offset(game_id: str) -> int:
    """Deterministic per-game integer offset added to EVAL_SEED.

    Python's built-in hash() is randomized per process, so it must never be
    used for seeding; this md5-based offset is stable everywhere.
    """
    return int(hashlib.md5(game_id.encode("utf-8")).hexdigest()[:8], 16)


def resolve_seed(game_id: str):
    """Return (seed, seed_source) per the contract.

    EVAL_SEED set   -> int(EVAL_SEED) + stable_game_offset(game_id), 'env'
    EVAL_SEED unset -> time-based fallback (NOT reproducible),       'time'

    The caller must seed every RNG library it uses (random, numpy, torch, ...)
    and run under PYTHONHASHSEED=0.
    """
    import time
    env_seed = os.getenv("EVAL_SEED")
    if env_seed is not None:
        return int(env_seed) + stable_game_offset(game_id), "env"
    return int(time.time() * 1_000_000) + stable_game_offset(game_id), "time"


def resolve_max_actions():
    """Return the action cap: float('inf') if EVAL_MAX_ACTIONS is unset/0."""
    cap = int(os.getenv("EVAL_MAX_ACTIONS", "0"))
    return cap if cap > 0 else float("inf")


def write_run_config(out_dir: str, **fields):
    """Write run_config.json recording exactly how this run was configured.

    Required fields for every agent: agent, game_id, seed, seed_source,
    max_actions. Add agent-specific extras freely.
    """
    cfg = dict(fields)
    if cfg.get("max_actions") == float("inf"):
        cfg["max_actions"] = None
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


class TransitionLogger:
    """Logs every (frame, action, next_frame, changed) transition to compressed
    .npz shards. Frames are stored as 64x64 uint8 color indices (NOT one-hot).

    Contract notes for implementers:
      - Log EVERY transition, before any agent-internal dedup/filtering.
      - `action_idx` uses the unified index: 0-4 = ACTION1-5,
        5 + (64*y + x) = click at (x, y).
      - `level` = the game score at the time of the action.
      - `model_ms` = your agent's decision-compute time for the action
        (0.0 is acceptable for trivial agents like random).
      - Call flush() on level change and register it with atexit.
    """

    def __init__(self, out_dir: str, flush_every: int = 1000):
        self.out_dir = out_dir
        self.flush_every = flush_every
        self.shard_idx = 0
        os.makedirs(out_dir, exist_ok=True)
        self._reset_buffers()

    def _reset_buffers(self):
        self.frames = []
        self.actions = []
        self.next_frames = []
        self.changed = []
        self.levels = []
        self.action_nums = []
        self.wall_ms = []
        self.model_ms = []

    def log(self, frame, action_idx, next_frame, changed, level, action_num,
            wall_ms, model_ms):
        self.frames.append(frame)
        self.actions.append(action_idx)
        self.next_frames.append(next_frame)
        self.changed.append(1 if changed else 0)
        self.levels.append(level)
        self.action_nums.append(action_num)
        self.wall_ms.append(wall_ms)
        self.model_ms.append(model_ms)
        if len(self.frames) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.frames:
            return
        path = os.path.join(self.out_dir, f"shard_{self.shard_idx:05d}.npz")
        np.savez_compressed(
            path,
            frames=np.stack(self.frames).astype(np.uint8),
            actions=np.array(self.actions, dtype=np.int32),
            next_frames=np.stack(self.next_frames).astype(np.uint8),
            changed=np.array(self.changed, dtype=np.uint8),
            levels=np.array(self.levels, dtype=np.int32),
            action_nums=np.array(self.action_nums, dtype=np.int64),
            wall_ms=np.array(self.wall_ms, dtype=np.float32),
            model_ms=np.array(self.model_ms, dtype=np.float32),
        )
        self.shard_idx += 1
        self._reset_buffers()
