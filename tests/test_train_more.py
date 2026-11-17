"""Unit tests for train_more.py (EVAL_TRAIN_EVERY, EVAL_TRAIN_RECENT).

    uv run python -m pytest tests/test_train_more.py -q
"""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_sampler import bare_agent
from train_more import TrainMore


def test_with_both_flags_at_todays_values_the_agent_holds_nothing(monkeypatch):
    monkeypatch.delenv("EVAL_TRAIN_EVERY", raising=False)
    monkeypatch.delenv("EVAL_TRAIN_RECENT", raising=False)
    assert TrainMore.from_env(5, 64) is None
    monkeypatch.setenv("EVAL_TRAIN_EVERY", "5")
    monkeypatch.setenv("EVAL_TRAIN_RECENT", "0")
    assert TrainMore.from_env(5, 64) is None
    assert bare_agent().train_more is None              # the class default: Goose's own training


def test_the_flags_are_read_and_bad_values_stop_the_run(monkeypatch):
    monkeypatch.setenv("EVAL_TRAIN_EVERY", "1")
    monkeypatch.setenv("EVAL_TRAIN_RECENT", "16")
    t = TrainMore.from_env(5, 64)
    assert (t.every, t.recent) == (1, 16) and t.config() == {"train_every": 1, "train_recent": 16}
    monkeypatch.setenv("EVAL_TRAIN_RECENT", "0")
    assert TrainMore.from_env(5, 64).recent == 0        # one flag alone is enough
    for every, recent in ((0, 0), (1, -1), (1, 65)):
        with pytest.raises(SystemExit):
            TrainMore(every, recent, 64)


def test_a_batch_holds_the_newest_moves_and_gooses_draw_for_the_rest():
    t = TrainMore(1, 16, 64)
    uniform = np.random.RandomState(0).choice(1000, 64, replace=False)
    b = t.batch(uniform, 1000)
    assert len(b) == 64 == len(set(b.tolist())) and b.dtype == uniform.dtype
    assert b[-16:].tolist() == list(range(984, 1000))                   # the 16 newest
    others = [i for i in uniform.tolist() if i < 984]
    assert b[:48].tolist() == others[:48]                               # Goose's draw, in its order
    assert t.steps == 1


def test_recent_zero_returns_gooses_batch_and_a_buffer_of_one_batch_is_used_whole():
    t = TrainMore(1, 0, 64)
    uniform = np.random.RandomState(1).choice(500, 64, replace=False)
    assert t.batch(uniform, 500) is uniform and t.steps == 1
    t = TrainMore(5, 16, 64)
    uniform = np.random.RandomState(2).choice(64, 64, replace=False)    # 64 stored: every one is drawn
    assert sorted(t.batch(uniform, 64).tolist()) == list(range(64))


def test_the_rest_of_the_batch_is_uniform_over_the_older_moves():
    t, rng = TrainMore(1, 4, 8), np.random.RandomState(0)
    counts = np.zeros(20)
    for _ in range(6000):
        b = t.batch(rng.choice(20, 8, replace=False), 20)
        counts[b] += 1
    assert (counts[16:] == 6000).all()                                  # the 4 newest: in every batch
    assert np.abs(counts[:16] / 6000 - 4 / 16).max() < 0.03             # each older move: 4 chances in 16


def test_the_agent_trains_on_the_newest_moves_when_asked():
    """The same check as test_sampler's batch test, with EVAL_TRAIN_RECENT=3."""
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
    a.train_more = TrainMore(1, 3, 8)
    rng = np.random.RandomState(0)
    raws = [rng.randint(0, 16, size=(64, 64)).astype(np.uint8) for _ in range(20)]
    a.experience_buffer = deque({"state": r, "action_idx": i, "reward": float(i % 2), "hash": i}
                                for i, r in enumerate(raws))
    np.random.seed(1)
    draw = np.random.choice(20, 8, replace=False)
    np.random.seed(1)                                   # the same draw inside the call
    a._train_action_model()
    want = [i for i in draw.tolist() if i < 17][:5] + [17, 18, 19]
    frames = torch.stack([a._frame_to_tensor(SimpleNamespace(frame=[raws[i]])) for i in want])
    assert torch.equal(a.action_model.seen, frames) and a.train_more.steps == 1
    assert np.random.random() == np.random.RandomState(1).permutation(20)[:0].sum() + after_one_draw(1, 20, 8)


def after_one_draw(seed, n, k):
    """The next np.random number after Goose's one batch draw: no second random call was made."""
    np.random.seed(seed)
    np.random.choice(n, k, replace=False)
    return np.random.random()
