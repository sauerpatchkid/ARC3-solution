"""Sampler-level tests for the agent's _sample_from_combined_output (the
degenerate-distribution guard: the tu93 crash) and for how a training batch is
built from the replay buffer.

Builds a bare Action object without running __init__ (no dirs, no
TensorBoard), which is enough to exercise the sampler.
"""
import logging
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

import run_local  # noqa: E402
run_local._install_minimal_agents_pkg()          # provides agents.structs
from agents.structs import GameAction  # noqa: E402
from action import Action  # noqa: E402


def bare_agent():
    a = Action.__new__(Action)
    a.grid_size = 64
    a.num_coordinates = 4096
    a._degenerate_samples = 0
    a.action_counter = 0
    a.logger = logging.getLogger("test")
    return a


ONLY_BUTTONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5]


def test_all_logits_underflow_falls_back_to_uniform_over_available():
    """Every sigmoid is exactly 0 -> old code raised 'probabilities contain
    NaN'. Now: uniform over the available actions, counter incremented."""
    a = bare_agent()
    logits = torch.full((4101,), -300.0)
    picks = set()
    for _ in range(50):
        idx, coords, cidx, _ = a._sample_from_combined_output(logits, ONLY_BUTTONS)
        assert coords is None and 0 <= idx < 5
        picks.add(idx)
    assert a._degenerate_samples == 50
    assert len(picks) >= 3, "uniform over the 5 buttons should hit several"


def test_guard_never_fires_on_healthy_logits():
    a = bare_agent()
    logits = torch.zeros(4101)
    for _ in range(20):
        a._sample_from_combined_output(logits, ONLY_BUTTONS)
    assert a._degenerate_samples == 0


def test_unavailable_actions_are_never_sampled():
    a = bare_agent()
    logits = torch.zeros(4101)
    avail = [GameAction.ACTION1, GameAction.ACTION3]   # no ACTION2, no clicks
    for _ in range(100):
        idx, coords, _, _ = a._sample_from_combined_output(logits, avail)
        assert coords is None and idx in (0, 2), idx


def test_training_batch_from_stored_frames_is_the_one_hot_the_network_sees():
    """The buffer stores 64x64 colour-index frames; the batch built from them
    must be exactly the one-hot tensors _frame_to_tensor makes for the same
    frames (what the buffer used to store)."""
    from collections import deque
    from types import SimpleNamespace

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            self.seen = x.detach().clone()
            return self.w.expand(x.size(0), 4101)

    a = bare_agent()
    a.device, a.num_colours, a.batch_size, a.log_metrics = torch.device("cpu"), 16, 8, False
    a.action_model = Capture()
    a.optimizer = torch.optim.SGD(a.action_model.parameters(), lr=0.1)
    rng = np.random.RandomState(0)
    raws = [rng.randint(0, 16, size=(64, 64)).astype(np.uint8) for _ in range(8)]
    a.experience_buffer = deque({"state": r, "action_idx": i, "reward": float(i % 2), "hash": i}
                                for i, r in enumerate(raws))
    np.random.seed(1)
    order = np.random.choice(8, 8, replace=False)
    np.random.seed(1)                                   # the same draw inside the call
    a._train_action_model()
    want = torch.stack([a._frame_to_tensor(SimpleNamespace(frame=[raws[i]])) for i in order])
    assert a.action_model.seen.dtype == torch.float32
    assert torch.equal(a.action_model.seen, want)
