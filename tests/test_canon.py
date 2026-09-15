"""Unit tests for Plan B's online canonicalizer and level memory (§7.1).

    uv run python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_agents"))

from canon import OnlineCanonicalizer, LevelMemory  # noqa: E402
from metrics_common import find_indicator_cells  # noqa: E402


# --------------------------------------------------------------------------
# Synthetic games: a static grid, one real object that moves only on demand,
# and one decoration - either a blinking cell (fixed ticker) or a 60-cell
# rotating bar (rotating ticker). Modelled separately on purpose: the shared
# detector only sees a rotating ticker when it ticks ALONE (<=2 cells changed
# in the whole transition), which is the known caveat in CLAUDE.md.
# --------------------------------------------------------------------------
def make_world():
    base = np.zeros((64, 64), dtype=np.uint8)
    base[10:20, 10:20] = 3          # static block
    return base


def frame(base, t, obj_x, blink=True, bar=True):
    f = base.copy()
    if blink:                        # fixed ticker: toggles every step
        f[0, 0] = 5 if t % 2 else 6
    if bar:                          # rotating ticker: one lit cell sweeping row 63
        f[63, t % 60] = 7
    f[40, obj_x] = 9                 # the real object
    return f


def play(canon, steps, move_at=(), blink=True, bar=True):
    base = make_world()
    obj_x = 30
    prev = frame(base, 0, obj_x, blink, bar)
    frames = [prev]
    for t in range(1, steps + 1):
        if t in move_at:
            obj_x += 1
        cur = frame(base, t, obj_x, blink, bar)
        canon.update(prev, cur)
        frames.append(cur)
        prev = cur
    return frames


def test_mask_all_false_before_warmup():
    c = OnlineCanonicalizer(warmup=200, refresh=250)
    play(c, 150)
    assert not c.mask.any()
    assert c.n_refreshes == 0


def test_fixed_ticker_masked_object_not():
    c = OnlineCanonicalizer(warmup=200, refresh=250)
    frames = play(c, 600, move_at=(300,), bar=False)
    assert c.mask[0, 0], "blinking cell is a fixed ticker"
    assert c.mask.sum() == 1
    assert not c.mask[40, 30] and not c.mask[40, 31], "the object is not decorative"
    assert c.key(frames[400]) == c.key(frames[401]), "blinking does not change the key"
    assert c.key(frames[299]) != c.key(frames[301]), "moving the object does"


def test_rotating_ticker_masked_object_not():
    c = OnlineCanonicalizer(warmup=200, refresh=250)
    frames = play(c, 600, move_at=(300,), blink=False)
    # The detector keeps the smallest cell set covering 95% of tiny-transition
    # changes, so with uniform counts it leaves ~5% of the bar unmasked (the
    # "ticker gaps" the archived LLM track filled by hand). Assert the rule as
    # specified, not an idealised 60/60.
    bar = c.mask[63, :60]
    assert bar.sum() >= 57, "the bar is a rotating ticker"
    assert c.mask.sum() == bar.sum(), "nothing outside the bar is masked"
    assert not c.mask[40, 30] and not c.mask[40, 31]
    # two consecutive frames whose lit bar cells are both masked: same key
    t = next(t for t in range(400, 460) if bar[t % 60] and bar[(t + 1) % 60])
    assert c.key(frames[t]) == c.key(frames[t + 1]), "the bar sweeping does not change the key"
    assert c.key(frames[299]) != c.key(frames[301]), "moving the object does"


@pytest.mark.xfail(strict=True, reason="known caveat (CLAUDE.md): the rotating-"
                   "ticker detector needs the ticker to tick alone; a bar that "
                   "always co-occurs with a blink is invisible to it")
def test_rotating_ticker_with_co_occurring_blink():
    c = OnlineCanonicalizer(warmup=200, refresh=250)
    play(c, 600)
    assert c.mask[63, :60].all()


def test_online_mask_matches_offline_on_same_counts():
    c = OnlineCanonicalizer(warmup=200, refresh=250)
    play(c, 1000)
    c.recompute_mask()
    freq = c.cell_change_count / c.n_trans
    offline = find_indicator_cells(freq=freq, tiny_frac=c.tiny_count / c.n_trans,
                                   tiny_cell_counts=c.tiny_cell_counts)
    assert np.array_equal(c.mask, offline)


def test_refresh_cadence():
    c = OnlineCanonicalizer(warmup=200, refresh=250)
    play(c, 1000)
    # once at warm-up (200), then at 250, 500, 750, 1000
    assert c.n_refreshes == 5


def test_key_is_raw_hash_before_warmup_and_stable():
    c = OnlineCanonicalizer()
    f = make_world()
    assert c.key(f) == c.key(f.copy())
    g = f.copy(); g[5, 5] = 1
    assert c.key(f) != c.key(g)


# --------------------------------------------------------------------------
# Label semantics and the tried-action mask
# --------------------------------------------------------------------------
def test_label_semantics_change_vs_novel():
    """Scripted sequence: no-op, change-to-new, change-back, change-to-new.
    change label = [0,1,1,1]; novel label = [0,1,0,1]."""
    mem = LevelMemory()
    A, B, C = 1, 2, 3
    mem.observe(A)                       # level start frame counts as seen
    seq = [A, B, A, C]
    prev = A
    change, novel = [], []
    for k in seq:
        change.append(int(k != prev))
        novel.append(int(mem.observe(k)))
        prev = k
    assert change == [0, 1, 1, 1]
    assert novel == [0, 1, 0, 1]


def test_mask_math_and_floor():
    mem = LevelMemory(decay=0.1, floor=1e-4)
    key = 42
    mem.record(key, 0)
    mem.record(key, 3); mem.record(key, 3)
    probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    out = mem.apply(probs, key)
    np.testing.assert_allclose(out, [0.05, 0.5, 0.5, 0.005, 0.5, 0.5], rtol=1e-6)
    # floor: after many tries the action is still possible
    for _ in range(10):
        mem.record(key, 0)
    out = mem.apply(probs, key)
    assert out[0] == pytest.approx(1e-4)
    # untried state: untouched (same object back)
    assert mem.apply(probs, 99) is probs


def test_unavailable_actions_stay_impossible():
    mem = LevelMemory(decay=0.1, floor=1e-4)
    mem.record(7, 2)
    probs = np.array([0.5, 0.5, 0.0, 0.5], dtype=np.float32)   # action 2 masked out
    out = mem.apply(probs, 7)
    assert out[2] == 0.0


def test_clear_resets_both_maps():
    mem = LevelMemory()
    mem.observe(1); mem.record(1, 0)
    mem.clear()
    assert mem.observe(1) is True
    assert mem.counts(1) == {}


def test_tried_cap_drops_oldest_half():
    mem = LevelMemory(max_states=4)
    for k in range(4):
        mem.record(k, 0)
    mem.record(100, 0)
    assert 0 not in mem.tried and 1 not in mem.tried
    assert 3 in mem.tried and 100 in mem.tried
