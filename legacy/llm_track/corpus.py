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
import collections
import glob
import json
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


class CorpusReader:
    """Random access to single transitions by global index.

    Loads the cheap scalar fields for the whole run up front (~25 B per
    transition), then decompresses a frame shard only when a transition inside
    it is requested, keeping a few in an LRU cache. Shards are NOT a fixed
    size — the logger also flushes on every level change — so indices are
    mapped through cumulative shard lengths.
    """

    def __init__(self, corpus_dir, cache_size=8):
        self.corpus_dir = corpus_dir
        self.paths = shard_paths(corpus_dir)
        fields = ("actions", "changed", "levels", "action_nums")
        cols = {k: [] for k in fields}
        lens = []
        for p in self.paths:
            with np.load(p) as z:
                for k in fields:
                    cols[k].append(z[k])
                lens.append(len(z["actions"]))
        self.scalars = {k: np.concatenate(v) for k, v in cols.items()}
        self.starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        self.cache_size = cache_size
        self._cache = collections.OrderedDict()

    def __len__(self):
        return int(self.starts[-1])

    @property
    def n_shards(self):
        return len(self.paths)

    def shard(self, s):
        """Frames of shard s (cached)."""
        if s in self._cache:
            self._cache.move_to_end(s)
            return self._cache[s]
        with np.load(self.paths[s]) as z:
            d = {"frames": z["frames"], "next_frames": z["next_frames"]}
        self._cache[s] = d
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return d

    def get(self, i):
        """One transition by global index."""
        i = int(i)
        s = int(np.searchsorted(self.starts, i, side="right")) - 1
        d = self.shard(s)
        k = i - int(self.starts[s])
        sc = self.scalars
        return {"frame": d["frames"][k], "next_frame": d["next_frames"][k],
                "action": int(sc["actions"][i]), "level": int(sc["levels"][i]),
                "action_num": int(sc["action_nums"][i]),
                "changed": bool(sc["changed"][i])}


def load_scalars(corpus_dir):
    """Load every non-frame field for a whole corpus (cheap: ~25 B/transition)."""
    fields = tuple(f for f in FIELDS if f not in ("frames", "next_frames"))
    out = {k: [] for k in fields}
    for s in iter_shards(corpus_dir, fields=fields):
        for k in fields:
            out[k].append(s[k])
    return {k: np.concatenate(v) for k, v in out.items()}


GOOSE = "stochastic_goose"      # the `agent` field Goose writes to run_config.json


def game_of(dirname):
    """'ft09-0d8bbf25' -> 'ft09'. API-path runs carry a version suffix; the
    game is the same, so anything keyed by game must normalise it."""
    return dirname.split("-")[0]


def run_agent(corpus_dir):
    """The agent that produced a corpus, from its run_config.json ('?' if the
    run predates run_config.json)."""
    p = os.path.join(os.path.dirname(corpus_dir.rstrip("/")), "run_config.json")
    try:
        with open(p) as f:
            return json.load(f).get("agent", "?")
    except (OSError, ValueError):
        return "?"


def find_corpora(results_dir="results", game=None, agent=None):
    """Return [(dirname, corpus_dir), ...] for every run under results/runs,
    oldest first. `agent` filters on run_config.json (e.g. GOOSE).

    `dirname` is the run's game directory, which may carry a version suffix —
    pass it through game_of() before using it as a game id.
    """
    pat = os.path.join(results_dir, "runs", "*", "*", "transitions")
    out = []
    for d in sorted(glob.glob(pat)):
        name = os.path.basename(os.path.dirname(d))
        if game is not None and game_of(name) != game:
            continue
        if agent is not None and run_agent(d) != agent:
            continue
        if not glob.glob(os.path.join(d, "shard_*.npz")):
            continue        # empty: a run that died before its first flush
        out.append((name, d))
    return out


def largest_per_game(corpora):
    """{game: corpus_dir}, choosing each game's run with the most shards
    (ties -> newest).

    Deliberately NOT "newest": a 2,000-action smoke test, or a teammate's
    agent, is often the newest run for a game, and anything computed from it —
    the ticker mask especially, see tickers.scan_corpus — would silently change.
    """
    best = {}
    for name, d in corpora:
        n = len(glob.glob(os.path.join(d, "shard_*.npz")))
        g = game_of(name)
        if n and (g not in best or n >= best[g][0]):
            best[g] = (n, d)
    return {g: d for g, (n, d) in best.items()}


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
