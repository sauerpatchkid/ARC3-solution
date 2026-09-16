"""Sampler-level tests for the agent's _sample_from_combined_output: the
tried-action mask and the degenerate-distribution guard (the tu93 crash).

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
from canon import LevelMemory  # noqa: E402


def bare_agent(mask_tried=False):
    a = Action.__new__(Action)
    a.grid_size = 64
    a.num_coordinates = 4096
    a.mask_tried = mask_tried
    a.memory = LevelMemory() if mask_tried else None
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


def test_underflow_with_mask_on_also_survives():
    a = bare_agent(mask_tried=True)
    for act in range(5):
        a.memory.record(7, act)          # every button tried once from this state
    logits = torch.full((4101,), -300.0)
    idx, coords, _, _ = a._sample_from_combined_output(logits, ONLY_BUTTONS, tried_key=7)
    assert coords is None and 0 <= idx < 5


def test_guard_never_fires_on_healthy_logits():
    a = bare_agent()
    logits = torch.zeros(4101)
    for _ in range(20):
        a._sample_from_combined_output(logits, ONLY_BUTTONS)
    assert a._degenerate_samples == 0


def test_mask_prefers_untried_button():
    a = bare_agent(mask_tried=True)
    key = 1
    for act in (0, 1, 2, 3):              # everything but ACTION5 tried 3x
        for _ in range(3):
            a.memory.record(key, act)
    logits = torch.zeros(4101)             # network indifferent
    counts = np.zeros(5, dtype=int)
    for _ in range(200):
        idx, coords, _, _ = a._sample_from_combined_output(logits, ONLY_BUTTONS, tried_key=key)
        counts[idx] += 1
    assert counts[4] > 150, counts        # ACTION5 dominates (others x0.001)


def test_unavailable_actions_never_sampled_under_mask():
    a = bare_agent(mask_tried=True)
    a.memory.record(3, 2)
    logits = torch.zeros(4101)
    avail = [GameAction.ACTION1, GameAction.ACTION3]   # ACTION2 not allowed
    for _ in range(100):
        idx, coords, _, _ = a._sample_from_combined_output(logits, avail, tried_key=3)
        assert idx in (0, 2), idx
