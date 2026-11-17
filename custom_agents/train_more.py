"""train_more.py — train Goose more often, and on fresher data.

THE PROBLEM IT FIXES
Goose takes one gradient step every 5 actions, on 64 stored moves drawn
uniformly from everything it has stored this level. With a couple of thousand
moves stored, a new one waits about 150 actions for its first update
(5 x 2,000 / 64). In a short run that is a large share of the budget spent
acting on a network that has not yet seen what the agent just found out.

WHAT IT DOES (two flags, each usable alone; the defaults are today's values)

  EVAL_TRAIN_EVERY   (default 5)  Train every N actions.
  EVAL_TRAIN_RECENT  (default 0)  This many of the newest stored moves go into
                                  every batch of 64; the rest are drawn
                                  uniformly from the others.

Nothing else about training changes: the batch size, the learning rate, the
loss and the buffer are Goose's. Why training more often should be safe here:
Goose already starts a new network at every level (EVAL_RESET_ON_LEVEL), and
periodic resets are what keep a high update rate from overfitting to early data
(the BBF result, Schwarzer et al. 2023).

How the batch is built. Goose's own draw is kept: 64 distinct stored moves,
uniformly at random, in random order (np.random.choice, replace=False). The
newest EVAL_TRAIN_RECENT moves are put in, and the batch is filled with the
first of Goose's draw that are not among them. The first so-many of a uniform
draw without replacement, with some members left out, are a uniform draw from
the rest, so no second random call is needed and Goose's random stream is used
exactly as often as before.

With both flags at their defaults `from_env` returns None, the agent holds
nothing from this module and trains exactly as before (checked move for move on
the CPU, tools/cpu_check.py).

HOW TO REMOVE IT
Leave both flags unset and nothing here runs. To delete it: this file,
tests/test_train_more.py, and the lines tagged `# [train]` in
custom_agents/action.py and check_repo.py (`sed -i '/\\[train\\]/d' <file>`
restores each byte for byte).

Its test: upgrade screen 3, docs/1-goose-upgrades/upgrade-screen3.md (arm T).

Pure numpy; no torch.
"""
import json
import os

import numpy as np


class TrainMore:
    def __init__(self, every, recent, batch_size):
        if every < 1:
            raise SystemExit(f"EVAL_TRAIN_EVERY must be 1 or more, got {every}")
        if not 0 <= recent <= batch_size:
            raise SystemExit(f"EVAL_TRAIN_RECENT must be between 0 and the batch size ({batch_size}), got {recent}")
        self.every, self.recent, self.batch_size = int(every), int(recent), int(batch_size)
        self.steps = 0              # gradient steps taken

    @classmethod
    def from_env(cls, default_every, batch_size):
        """The two flags, or None when both are at today's values (so that an
        unflagged run holds nothing from this module)."""
        every = int(os.getenv("EVAL_TRAIN_EVERY", default_every))
        recent = int(os.getenv("EVAL_TRAIN_RECENT", 0))
        if every == default_every and recent == 0:
            return None
        return cls(every, recent, batch_size)

    def batch(self, uniform, n):
        """The batch to train on. `uniform` is Goose's own draw: distinct indices
        into its `n` stored moves (0 = oldest), uniformly at random, in random
        order. Returns as many indices: the `recent` newest, and before them the
        first of `uniform` that are not among those (module doc)."""
        self.steps += 1
        k = min(self.recent, len(uniform))
        if k == 0:
            return uniform
        uniform = np.asarray(uniform)
        others = uniform[uniform < n - k][:len(uniform) - k]
        return np.concatenate([others, np.arange(n - k, n, dtype=uniform.dtype)])

    def config(self):
        """Fields for run_config.json."""
        return {"train_every": self.every, "train_recent": self.recent}

    def summary(self):
        return dict(self.config(), train_steps=self.steps, batch_size=self.batch_size)

    def dump(self, path):
        try:
            with open(path, "w") as f:
                json.dump(self.summary(), f, indent=2)
        except OSError:
            pass
