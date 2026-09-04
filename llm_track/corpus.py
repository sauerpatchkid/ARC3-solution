"""corpus.py — shard iteration helpers shared by every offline LLM-track tool.

The transition corpus is written by eval_common.TransitionLogger as compressed
.npz shards; see that module for the field contract. Everything here is read-
only, so no tool in this package can perturb a run.

Two facts about the corpus that callers must not forget:

  * `action_nums` is NOT contiguous. The agent drops a transition whenever
    prev_frame is cleared: at a game over (~1.5% of actions on ft09, two
    action_nums lost each time) and, in corpora recorded before the
    level-boundary logging fix, at every level-up. Always align on
    `action_nums`, never on array position.

  * `levels` is the score at the time the action was TAKEN. A level-completing
    transition therefore carries the OLD level, and the level-up is visible as
    levels[i] < levels[i+1]. Corpora recorded before the fix are missing that
    transition entirely (there, the gap in action_nums is the only trace).
"""
import glob
import os

import numpy as np

FIELDS = ("frames", "actions", "next_frames", "changed", "levels",
          "action_nums", "wall_ms", "model_ms")


def shard_paths(corpus_dir):
    """Sorted shard paths for a transitions/ directory."""
    paths = sorted(glob.glob(os.path.join(corpus_dir, "shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no shard_*.npz in {corpus_dir}")
    return paths


def iter_shards(corpus_dir, fields=None, limit=None):
    """Yield one dict of arrays per shard.

    `fields` restricts what is decompressed — passing only what you need is a
    large win, since `frames`/`next_frames` are 4 KB per transition and the
    other six fields together are 25 bytes.
    """
    paths = shard_paths(corpus_dir)
    if limit is not None:
        paths = paths[:limit]
    for p in paths:
        with np.load(p) as z:
            keys = fields if fields is not None else z.files
            yield {k: z[k] for k in keys}


def load_scalars(corpus_dir):
    """Load every non-frame field for a whole corpus (cheap: ~25 B/transition)."""
    fields = tuple(f for f in FIELDS if f not in ("frames", "next_frames"))
    out = {k: [] for k in fields}
    for s in iter_shards(corpus_dir, fields=fields):
        for k in fields:
            out[k].append(s[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def find_corpora(results_dir="results", game=None):
    """Return [(game_id, corpus_dir), ...] for every run under results/runs."""
    pat = os.path.join(results_dir, "runs", "*", game or "*", "transitions")
    out = []
    for d in sorted(glob.glob(pat)):
        out.append((os.path.basename(os.path.dirname(d)), d))
    return out


def level_events(levels, action_nums):
    """Return [(index, action_num, from_level, to_level), ...] for each level-up.

    `index` is the position of the LAST transition at the old level. Post-fix
    corpora make that the level-completing transition itself; pre-fix corpora
    make it the one before it (the completing transition was never logged).
    Use `is_post_fix_boundary` to tell the two apart.
    """
    idx = np.where(np.diff(levels) > 0)[0]
    return [(int(i), int(action_nums[i]), int(levels[i]), int(levels[i + 1]))
            for i in idx]


def is_post_fix_boundary(action_nums, i):
    """True if the level-completing transition at boundary index `i` was logged.

    Post-fix the completing transition is present, so consecutive action_nums
    differ by 1; pre-fix it is missing and they differ by 2.
    """
    return int(action_nums[i + 1]) - int(action_nums[i]) == 1
