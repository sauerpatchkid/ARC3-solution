"""rule_referee.py — the checker for Qwen-written game rules (rule-finding, Stage A).

A RULE is two small functions Qwen writes (RULE_DOC below is the exact text it
sees):

    RULE = "one line: what happens"
    def applies(board, act, api):    # does this rule claim to know what happens?
    def predict(board, act, api):    # the 64x64 board right after the action

THE CHECK. On every recorded move where applies() is True, predict() must
reproduce the real next board EXACTLY on every non-ticker cell. A rule is
ACCEPTED if it applies to at least 20 training moves and is exactly right on at
least 95% of them. A RULE BOOK predicts a move with its first rule that applies,
or "nothing changes" if none does, so a rule book with no rules scores exactly
the "nothing" baseline.

DATA (fixed before any rule-writing run):
  ft09, lp85   the loop sees LEVEL 1 only; the test set is LEVEL 2.
  ls20         never reaches level 2, so its runs are split in two, alternating
               by run id: the loop sees one half, the test is the other half.
  Moves on a FROZEN screen are dropped: a stretch of more than 500 moves in a
  row that change nothing at all, ticker included. The game has stopped
  responding (three ft09 runs sit on one frozen screen for 25k-199k moves), so
  there is no mechanic to learn, and those moves would swamp any sample.
  The ticker mask is tickers.py's detected ticker with the gaps inside each of
  its rows filled. A few progress-bar cells change too rarely to be detected
  (ft09 row 63: 2 cells; ls20 row 61: 3), and on ft09 they alone kept a correct
  tile-flip rule at 94%: all 35 of its misses were one such cell.
  Samples are spread as evenly over runs as the pool allows, so each run counts
  about equally (the CIs are bootstrapped over runs). Test moves are otherwise
  uniform - the natural mix of actions. Training moves are then sampled per
  action group (each button; clicks grouped by the colour clicked) so rare
  actions still get examples.
  Moves whose next board is the NEXT level's first screen are dropped: no rule
  about mechanics can predict a level transition.

BASELINES (no LLM), learned from the TRAINING moves only:
  nothing   the board does not change.
  memory    if this exact board + action was seen in training, repeat what
            happened then; else, for a click, if the same 9x9 neighbourhood was
            clicked in training, repeat that change (within 8 cells) around the
            new click; else nothing changes. This is the honest "do we need an
            LLM?" bar: it memorises well and generalises only locally.

    uv run python -m legacy.llm_track.rule_referee build      # -> results/legacy_llm/rules/data/
    uv run python -m legacy.llm_track.rule_referee selftest   # controls, baselines, sandbox
"""
import argparse
import collections
import json
import os
import subprocess
import tempfile
import time

import numpy as np

from .corpus import GOOSE, CorpusReader, find_corpora, game_of, is_post_fix_boundary
from .heur_api import HeuristicAPI, background_of, regions
from .heur_sandbox import compile_functions, run_in_child, validate
from .probe_images import PALETTE_NAMES
from .probe_pairs import build_masks

DATA_DIR = "results/legacy_llm/rules/data"
GAMES = ("ft09", "lp85", "ls20")
SPLIT = {"ft09": "level", "lp85": "level", "ls20": "runs"}
TRAIN_PER_GROUP = 800        # training moves kept per action group
TRAIN_CANDIDATES = 25000     # uniform pre-sample from which groups are formed
TEST_MOVES = 4000            # uniform sample of the test pool
MIN_GROUP = 30               # smaller groups are not worth a prompt
FROZEN = 500                 # longer stretches of no-change-at-all moves are a frozen screen
PATCH, WINDOW = 4, 8         # memory baseline: 9x9 key, change within 8 cells
ACCEPT = {"min_applies": 20, "min_accuracy": 0.95, "max_mean_ms": 5.0}
N_WRONG = 12                 # mistakes kept per checked rule (a random sample), to show Qwen
RULE_FUNCS = {"applies": 3, "predict": 3}

RULE_DOC = """You write ONE rule about how this game works, as Python. `np` (numpy) is \
available; nothing can be imported.

    RULE = "one line saying what happens"
    def applies(board, act, api):
        # True if your rule knows what this action does on this board
        ...
    def predict(board, act, api):
        # the board right AFTER the action
        out = board.copy()
        ...
        return out

`board` is the board BEFORE the action: a 64x64 numpy array of colour indices 0-15.
It is read-only, so work on board.copy().
`act.action` is 1-5 for a button (ACTION1-ACTION5) or 6 for a click; `act.click`
is (row, col) for a click and None otherwise.
`api` gives you:
  api.first       the board at the START of this level (64x64)
  api.background  the most common colour of api.first
  api.ticker      64x64 bool: timer/counter cells that change by themselves. They
                  are ignored when your rule is checked.
  api.layout      regions on api.first: connected same-colour areas (background and
                  ticker excluded), largest first. Each region r has r.colour,
                  r.size, r.y0, r.x0, r.y1, r.x1 (inclusive bounding box), r.mask
                  (64x64 bool, True on the region) and r.cells (n x 2 (row, col)).

How rules are checked: on every recorded move where applies() is True, predict()
must reproduce the real next board EXACTLY on every non-ticker cell. A rule is kept
if it applies to at least 20 recorded moves and is exactly right on at least 95%
of them. So:
- make applies() narrow enough that the rule is right whenever it says it applies;
- any move no rule covers is predicted as "nothing changes", so a rule is only worth
  something if it predicts the CHANGES exactly; "this does nothing" rules add nothing;
- it is tested on a DIFFERENT level (or different runs) of this game, so describe
  the mechanism, not specific positions;
- keep it fast: whole-array numpy, no Python loops over all 4,096 cells.

Example, from a DIFFERENT game - it shows the style, not the answer here:

    RULE = "ACTION1 moves the green (14) block up by one row"
    def applies(board, act, api):
        return act.action == 1 and (board == 14).any()
    def predict(board, act, api):
        out = board.copy()
        g = board == 14
        out[g] = api.background
        out[np.roll(g, -1, axis=0)] = 14
        return out"""


class Act:
    """The `act` argument: what the player did."""
    __slots__ = ("action", "click")

    def __init__(self, action_idx):
        a = int(action_idx)
        if a < 5:
            self.action, self.click = a + 1, None
        else:
            self.action, self.click = 6, divmod(a - 5, 64)

    def __repr__(self):
        return f"Act(action={self.action}, click={self.click})"


def group_of(action_idx, before):
    """Action group: each button, and clicks by the colour clicked."""
    a = int(action_idx)
    if a < 5:
        return f"ACTION{a + 1}"
    y, x = divmod(a - 5, 64)
    c = int(before[y, x])
    return f"click on colour {c} ({PALETTE_NAMES[c]})"


def _readonly(a):
    a = np.asarray(a)
    a.flags.writeable = False
    return a


# ------------------------------------------------------------------ building
def _frozen(changed):
    """True for moves inside a stretch of more than FROZEN consecutive moves that
    changed nothing on screen (ticker included)."""
    x = np.diff(np.r_[True, np.asarray(changed, bool), True].astype(np.int8))
    out = np.zeros(len(changed), bool)
    for s, e in zip(np.flatnonzero(x == -1), np.flatnonzero(x == 1)):
        if e - s > FROZEN:
            out[s:e] = True
    return out


def _fill_rows(mask):
    """The ticker mask with the gaps inside each of its rows filled."""
    out = mask.copy()
    for y in np.flatnonzero(mask.any(axis=1)):
        xs = np.flatnonzero(mask[y])
        out[y, xs.min():xs.max() + 1] = True
    return out


def _balanced(pool, n, rng):
    """About n moves from pool, spread as evenly over runs as their sizes allow."""
    by_run = collections.defaultdict(list)
    for item in pool:
        by_run[item[0]].append(item)
    lo, hi = 0, max(len(v) for v in by_run.values())
    while lo < hi:                                  # smallest per-run cap giving n moves
        c = (lo + hi) // 2
        if sum(min(len(v), c) for v in by_run.values()) >= n:
            hi = c
        else:
            lo = c + 1
    out = []
    for ri in sorted(by_run):
        v = by_run[ri]
        out += [v[j] for j in rng.choice(len(v), min(len(v), lo), replace=False)]
    return out


def _pool(readers, runs, level):
    """(reader index, transition index) for every usable move at `level`, and how
    many were dropped as frozen."""
    out, n_frozen = [], 0
    for ri in runs:
        r = readers[ri]
        lv, an = r.scalars["levels"], r.scalars["action_nums"]
        frozen = _frozen(r.scalars["changed"])
        n_frozen += int((frozen & (lv == level)).sum())
        idx = np.flatnonzero((lv == level) & ~frozen)
        # a logged level-completing move's next board is the next level's screen
        edges = np.flatnonzero(np.diff(lv) > 0)
        drop = {int(e) for e in edges if is_post_fix_boundary(an, e)}
        out += [(ri, int(i)) for i in idx if int(i) not in drop]
    return out, n_frozen


def _segment_starts(levels):
    """First index of every contiguous same-level stretch of a run."""
    return np.r_[0, np.flatnonzero(np.diff(levels) != 0) + 1]


def build(results="results", out=DATA_DIR, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    corpora = find_corpora(results, agent=GOOSE)
    masks, mask_info, _ = build_masks([c for c in corpora if game_of(c[0]) in GAMES])
    filled = {g: int(_fill_rows(m).sum() - m.sum()) for g, m in masks.items()}
    masks = {g: _fill_rows(m) for g, m in masks.items()}
    os.makedirs(out, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None

    for g in GAMES:
        paths = [d for n, d in corpora if game_of(n) == g]
        readers = [CorpusReader(d) for d in paths]
        if SPLIT[g] == "runs":                        # ls20: alternate runs
            train_runs, test_runs = list(range(0, len(paths), 2)), list(range(1, len(paths), 2))
            train_level = test_level = 0
        else:                                         # level 1 -> level 2
            train_runs = test_runs = list(range(len(paths)))
            train_level, test_level = 0, 1
        train_pool, train_frozen = _pool(readers, train_runs, train_level)
        test_pool, test_frozen = _pool(readers, test_runs, test_level)
        pre = _balanced(train_pool, TRAIN_CANDIDATES, rng)
        test = _balanced(test_pool, TEST_MOVES, rng)

        def load(items):
            rows = []
            for ri, i in sorted(items):
                t = readers[ri].get(i)
                rows.append((ri, i, t["frame"].copy(), t["next_frame"].copy(), int(t["action"]),
                             int(readers[ri].scalars["levels"][i])))
            return rows

        by_group = collections.defaultdict(list)
        for row in load(pre):
            by_group[group_of(row[4], row[2])].append(row)
        groups = sorted((k for k, v in by_group.items() if len(v) >= MIN_GROUP),
                        key=lambda k: -len(by_group[k]))
        train_rows = []
        for k in groups:
            rows = by_group[k]
            keep = rng.choice(len(rows), min(TRAIN_PER_GROUP, len(rows)), replace=False)
            train_rows += [rows[j] for j in sorted(keep)]
        test_rows = load(test)

        seg = {ri: _segment_starts(r.scalars["levels"]) for ri, r in enumerate(readers)}
        firsts, first_key = [], {}
        def first_index(ri, i):
            s0 = int(seg[ri][np.searchsorted(seg[ri], i, side="right") - 1])
            key = (ri, s0)
            if key not in first_key:
                first_key[key] = len(firsts)
                firsts.append(readers[ri].get(s0)["frame"].copy())
            return first_key[key]

        rows = [(0, r) for r in train_rows] + [(1, r) for r in test_rows]
        gidx = {k: n for n, k in enumerate(groups)}
        np.savez_compressed(
            os.path.join(out, f"{g}.npz"),
            before=np.stack([r[2] for _, r in rows]).astype(np.uint8),
            after=np.stack([r[3] for _, r in rows]).astype(np.uint8),
            actions=np.array([r[4] for _, r in rows], np.int32),
            split=np.array([s for s, _ in rows], np.int8),
            group=np.array([gidx.get(group_of(r[4], r[2]), -1) for _, r in rows], np.int16),
            run=np.array([r[0] for _, r in rows], np.int16),
            level=np.array([r[5] for _, r in rows], np.int16),
            first_idx=np.array([first_index(r[0], r[1]) for _, r in rows], np.int32),
            firsts=np.stack(firsts).astype(np.uint8),
            mask=masks[g])
        counts = collections.Counter((s, group_of(r[4], r[2])) for s, r in rows)
        meta = {"game": g, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit,
                "split": SPLIT[g], "runs": paths, "train_runs": train_runs, "test_runs": test_runs,
                "train_level": train_level, "test_level": test_level, "groups": groups,
                "n_train": len(train_rows), "n_test": len(test_rows),
                "train_per_group": {k: counts[(0, k)] for k in groups},
                "test_per_group": {k: counts[(1, k)] for k in
                                   sorted({group_of(r[4], r[2]) for r in test_rows})},
                "pool": {"train": len(train_pool), "test": len(test_pool)},
                "frozen_dropped": {"train": train_frozen, "test": test_frozen},
                "mask": mask_info.get(g), "mask_holes_filled": filled.get(g, 0), "seed": seed}
        with open(os.path.join(out, f"{g}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[rules] {g}: {len(train_rows)} training moves in {len(groups)} groups, "
              f"{len(test_rows)} test moves ({SPLIT[g]} split); frozen moves dropped: "
              f"{train_frozen} train, {test_frozen} test  [{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------ data access
class RuleData:
    """One game's rule dataset, as the checker and the writer see it."""

    def __init__(self, game, d=DATA_DIR):
        z = np.load(os.path.join(d, f"{game}.npz"))
        self.game = game
        self.before, self.after = z["before"], z["after"]
        self.actions, self.split, self.group = z["actions"], z["split"], z["group"]
        self.run, self.level, self.first_idx = z["run"], z["level"], z["first_idx"]
        self.firsts = z["firsts"]
        self.mask = _readonly(z["mask"].copy())
        self.live = ~self.mask
        self.changed = ((self.before != self.after) & self.live).reshape(len(self.actions), -1).any(axis=1)
        self.meta = json.load(open(os.path.join(d, f"{game}.json")))
        self.groups = self.meta["groups"]
        self._layout = {}

    def idx(self, split, group=None):
        m = self.split == (0 if split == "train" else 1)
        if group is not None:
            m &= self.group == self.groups.index(group)
        return np.flatnonzero(m)

    def inputs(self, i):
        """(board, act, api), read-only, exactly as a live agent would build them."""
        f = int(self.first_idx[i])
        if f not in self._layout:
            first = _readonly(self.firsts[f].copy())
            lay = regions(first, background_of(first), self.mask)
            for reg in lay:
                reg.cells.flags.writeable = False
                reg.mask.flags.writeable = False
            self._layout[f] = (first, lay)
        first, lay = self._layout[f]
        api = HeuristicAPI(first, self.mask, layout=lay, level=int(self.level[i]))
        return _readonly(self.before[i].copy()), Act(self.actions[i]), api

    def exact(self, i, pred):
        pred = np.asarray(pred)
        return pred.shape == (64, 64) and bool(np.array_equal(pred[self.live], self.after[i][self.live]))


# ------------------------------------------------------------------ checking rules
def _rule_worker(src, game, d, split, max_moves, seed, q):
    """Child process: run one rule over a split's moves."""
    try:
        ns = compile_functions(src, RULE_FUNCS)
        data = RuleData(game, d)
        idx = data.idx(split)
        if max_moves and len(idx) > max_moves:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_moves, replace=False))
        applies = np.zeros(len(idx), bool)
        correct = np.zeros(len(idx), bool)
        ms = np.zeros(len(idx))
        errors, first_error = 0, None
        rng, wrong, n_wrong = np.random.default_rng(seed + 1), [], 0
        for k, i in enumerate(idx):
            board, act, api = data.inputs(i)
            t0 = time.perf_counter()
            try:
                if ns["applies"](board, act, api):
                    pred = np.asarray(ns["predict"](board, act, api))
                    if pred.shape != (64, 64):
                        raise ValueError(f"predict must return a 64x64 board, got shape {pred.shape}")
                    applies[k] = True
                    correct[k] = data.exact(i, pred)
                    if not correct[k]:                 # reservoir sample of the mistakes
                        n_wrong += 1
                        keep = (int(i), np.clip(pred, 0, 15).astype(np.uint8))
                        if len(wrong) < N_WRONG:
                            wrong.append(keep)
                        elif (j := int(rng.integers(0, n_wrong))) < N_WRONG:
                            wrong[j] = keep
            except Exception as e:
                errors += 1
                first_error = first_error or f"{type(e).__name__}: {e}"
                if errors > 0.05 * len(idx) + 5:           # systematic, not a rare edge case
                    raise RuntimeError(first_error)
            ms[k] = (time.perf_counter() - t0) * 1000.0
        q.put({"ok": True, "rule": str(ns.get("RULE", "")), "idx": idx, "applies": applies,
               "correct": correct, "ms": ms, "errors": errors, "first_error": first_error,
               "wrong": wrong})
    except Exception as e:
        q.put({"ok": False, "stage": "runtime", "reason": f"{type(e).__name__}: {e}"})


def check_rule(src, game, d=DATA_DIR, split="train", max_moves=None, timeout_s=300, seed=0):
    """Validate, run in a child process, and score one rule.

    Returns a dict: ok, stage ('static' | 'runtime' | 'timeout' | 'checked'),
    reason, and when checked: rule, n, n_applies, coverage, n_correct, accuracy,
    accepted, ms_mean, errors, plus per-move arrays idx / applies / correct.
    """
    ok, why = validate(src, RULE_FUNCS)
    if not ok:
        return {"ok": False, "stage": "static", "reason": why}
    res = run_in_child(_rule_worker, (src, game, d, split, max_moves, seed), timeout_s)
    if not res["ok"]:
        return res
    n, na, nc = len(res["idx"]), int(res["applies"].sum()), int(res["correct"].sum())
    ms = float(res["ms"].mean()) if n else 0.0
    acc = nc / na if na else 0.0
    accepted = (na >= ACCEPT["min_applies"] and acc >= ACCEPT["min_accuracy"]
                and ms <= ACCEPT["max_mean_ms"])
    reason = "ok" if accepted else (
        f"applies to only {na} moves (needs {ACCEPT['min_applies']})" if na < ACCEPT["min_applies"] else
        f"right on {acc:.0%} of the {na} moves it applies to (needs {ACCEPT['min_accuracy']:.0%})"
        if acc < ACCEPT["min_accuracy"] else f"too slow: {ms:.1f} ms per move")
    return {**res, "ok": True, "stage": "checked", "reason": reason, "n": n, "n_applies": na,
            "coverage": round(na / n, 4) if n else 0.0, "n_correct": nc,
            "accuracy": round(acc, 4), "accepted": accepted, "ms_mean": round(ms, 3)}


def _book_worker(srcs, game, d, split, q):
    """Child process: predict every move with a rule book (first rule that applies)."""
    try:
        books = [compile_functions(s, RULE_FUNCS) for s in srcs]
        data = RuleData(game, d)
        idx = data.idx(split)
        correct = np.zeros(len(idx), bool)
        used = np.full(len(idx), -1, np.int16)
        for k, i in enumerate(idx):
            board, act, api = data.inputs(i)
            pred = board
            for j, ns in enumerate(books):
                try:
                    if ns["applies"](board, act, api):
                        pred, used[k] = ns["predict"](board, act, api), j
                        break
                except Exception:
                    continue
            correct[k] = data.exact(i, pred)
        q.put({"ok": True, "idx": idx, "correct": correct, "used": used})
    except Exception as e:
        q.put({"ok": False, "stage": "runtime", "reason": f"{type(e).__name__}: {e}"})


def rule_book(srcs, game, d=DATA_DIR, split="test", timeout_s=600):
    return run_in_child(_book_worker, (list(srcs), game, d, split), timeout_s)


# ------------------------------------------------------------------ baselines
def baseline_nothing(data, idx):
    return np.array([np.array_equal(data.before[i][data.live], data.after[i][data.live])
                     for i in idx])


def _patch(board, y, x):
    padded = np.pad(board, PATCH, constant_values=255)
    return padded[y:y + 2 * PATCH + 1, x:x + 2 * PATCH + 1].tobytes()


def _delta(before, after, y, x, live):
    y0, y1, x0, x1 = max(0, y - WINDOW), min(64, y + WINDOW + 1), max(0, x - WINDOW), min(64, x + WINDOW + 1)
    ch = np.argwhere((before[y0:y1, x0:x1] != after[y0:y1, x0:x1]) & live[y0:y1, x0:x1])
    return tuple((int(r + y0 - y), int(c + x0 - x), int(after[r + y0, c + x0])) for r, c in ch)


def baseline_memory(data, train_idx, idx):
    live = data.live
    exact = {}
    local = collections.defaultdict(collections.Counter)
    for i in train_idx:
        b, a, act = data.before[i], data.after[i], int(data.actions[i])
        exact[(np.where(live, b, 255).tobytes(), act)] = i
        if act >= 5:
            y, x = divmod(act - 5, 64)
            local[_patch(b, y, x)][_delta(b, a, y, x, live)] += 1
    ok = np.zeros(len(idx), bool)
    for k, i in enumerate(idx):
        b, a, act = data.before[i], data.after[i], int(data.actions[i])
        key = (np.where(live, b, 255).tobytes(), act)
        if key in exact:
            pred = data.after[exact[key]]
        elif act >= 5 and _patch(b, *divmod(act - 5, 64)) in local:
            y, x = divmod(act - 5, 64)
            pred = b.copy()
            for dy, dx, c in local[_patch(b, y, x)].most_common(1)[0][0]:
                if 0 <= y + dy < 64 and 0 <= x + dx < 64:
                    pred[y + dy, x + dx] = c
        else:
            pred = b
        ok[k] = np.array_equal(pred[live], a[live])
    return ok


def compare(data, idx, correct, n_boot=2000, seed=0):
    """Exact-prediction rates with 95% CIs bootstrapped over RUNS (moves from one
    run are not independent), plus paired CIs for every difference vs the first
    entry of `correct` (a dict name -> bool array aligned with idx)."""
    runs = data.run[idx]
    groups = [np.flatnonzero(runs == r) for r in np.unique(runs)]
    rng = np.random.default_rng(seed)
    names = list(correct)
    boot = {n: [] for n in names}
    for _ in range(n_boot):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for n in names:
            boot[n].append(correct[n][sel].mean())
    out = {n: {"rate": round(float(correct[n].mean()), 4),
               "ci": [round(float(v), 4) for v in np.percentile(boot[n], [2.5, 97.5])]}
           for n in names}
    first = names[0]
    for n in names[1:]:
        diff = np.array(boot[first]) - np.array(boot[n])
        out[n]["diff_vs_" + first] = [round(float(v), 4) for v in np.percentile(diff, [2.5, 97.5])]
    return out, len(groups)


# ------------------------------------------------------------------ self-test
NOTHING_RULE = '''RULE = "nothing ever changes"
def applies(board, act, api):
    return True
def predict(board, act, api):
    return board.copy()
'''

# A HAND-WRITTEN rule for ft09 - NOT an LLM result, and only partly right (on
# level 1 some clicked tiles do not flip, and some flips change one extra cell).
# Its job is to show that the click decoding matches the recordings: it must
# predict many real tile flips that a row/col-swapped reading gets wrong, and
# hardly any the other way round.
FT09_FLIP_RULE = '''RULE = "clicking a red or blue cell flips its whole tile between red and blue"
def applies(board, act, api):
    if act.click is None:
        return False
    y, x = act.click
    return board[y, x] == 8 or board[y, x] == 9
def predict(board, act, api):
    y, x = act.click
    out = board.copy()
    for r in api.layout:
        if r.mask[y, x]:
            out[r.mask] = 9 if board[y, x] == 8 else 8
            break
    return out
'''

# A synthetic game whose mechanics are KNOWN, to test the checker itself: ACTION4
# moves the orange (12) block right by 2, clicking the block turns it blue (9),
# and ACTION1 changes nothing.
TOY_MOVE_RULE = '''RULE = "ACTION4 moves the orange block right by 2"
def applies(board, act, api):
    return act.action == 4
def predict(board, act, api):
    out = board.copy()
    m = board == 12
    out[m] = 0
    out[np.roll(m, 2, axis=1)] = 12
    return out
'''

TOY_CLICK_RULE = '''RULE = "clicking the orange block turns it blue"
def applies(board, act, api):
    return act.click is not None and board[act.click[0], act.click[1]] == 12
def predict(board, act, api):
    out = board.copy()
    out[board == 12] = 9
    return out
'''


def _toy_data(d, n=300):
    """Write the synthetic game in the rules-dataset format; returns its actions."""
    rng = np.random.default_rng(0)
    before, after, actions = [], [], []
    for _ in range(n):
        b = np.zeros((64, 64), np.uint8)
        b[40:44, 5:60] = 2
        y, x = int(rng.integers(0, 36)), int(rng.integers(0, 56))
        b[y:y + 4, x:x + 4] = 12
        a, kind = b.copy(), int(rng.integers(0, 3))
        if kind == 0:
            act = 3
            a[y:y + 4, x:x + 4] = 0
            a[y:y + 4, x + 2:x + 6] = 12
        elif kind == 1:
            act = 5 + 64 * (y + int(rng.integers(0, 4))) + x + int(rng.integers(0, 4))
            a[y:y + 4, x:x + 4] = 9
        else:
            act = 0
        before.append(b), after.append(a), actions.append(act)
    np.savez_compressed(os.path.join(d, "toy.npz"), before=np.stack(before), after=np.stack(after),
                        actions=np.array(actions, np.int32), split=np.zeros(n, np.int8),
                        group=np.zeros(n, np.int16), run=np.zeros(n, np.int16),
                        level=np.zeros(n, np.int16), first_idx=np.zeros(n, np.int32),
                        firsts=np.stack(before[:1]), mask=np.zeros((64, 64), bool))
    with open(os.path.join(d, "toy.json"), "w") as f:
        json.dump({"game": "toy", "groups": ["all"], "train_per_group": {"all": n}}, f)
    return np.array(actions)


BAD_RULES = {
    "imports a module": ("import os\ndef applies(board, act, api):\n    return True\n"
                         "def predict(board, act, api):\n    return board.copy()", "static"),
    "missing predict": ("def applies(board, act, api):\n    return True", "static"),
    "wrong board shape": ("def applies(board, act, api):\n    return True\n"
                          "def predict(board, act, api):\n    return np.zeros(5)", "runtime"),
}


def selftest(d=DATA_DIR):
    ok_all = True
    print("1. Data and the two no-LLM baselines on each game's TEST set:")
    for g in GAMES:
        data = RuleData(g, d)
        tr, te = data.idx("train"), data.idx("test")
        res, n_runs = compare(data, te, {"nothing": baseline_nothing(data, te),
                                          "memory": baseline_memory(data, tr, te)})
        print(f"   {g}: {len(tr)} train / {len(te)} test moves ({n_runs} test runs), groups: "
              f"{', '.join(f'{k} ({v})' for k, v in data.meta['train_per_group'].items())}")
        for n, r in res.items():
            print(f"      {n:<8} exact-prediction rate {r['rate']:.3f}  CI {r['ci'][0]:.3f}-{r['ci'][1]:.3f}")

    print("\n2. Controls, run through the sandbox:")
    for g in GAMES:
        data = RuleData(g, d)
        te = data.idx("test")
        book = rule_book([NOTHING_RULE], g, d)
        same = book["ok"] and bool((book["correct"] == baseline_nothing(data, te)).all())
        ok_all &= same
        print(f"   [{'PASS' if same else 'FAIL'}] {g}: a 'nothing changes' rule book equals the "
              f"nothing baseline exactly")
    with tempfile.TemporaryDirectory() as td:
        acts = _toy_data(td)
        n4, nc = int((acts == 3).sum()), int((acts >= 5).sum())
        swapped = TOY_CLICK_RULE.replace("board[act.click[0], act.click[1]]",
                                         "board[act.click[1], act.click[0]]")
        for name, src, want in (
                ("toy: correct move rule", TOY_MOVE_RULE,
                 lambda r: r["n_applies"] == n4 and r["accuracy"] == 1.0),
                ("toy: move rule off by one", TOY_MOVE_RULE.replace("np.roll(m, 2,", "np.roll(m, 1,"),
                 lambda r: r["n_applies"] == n4 and r["accuracy"] == 0.0),
                ("toy: correct click rule", TOY_CLICK_RULE,
                 lambda r: r["n_applies"] == nc and r["accuracy"] == 1.0),
                ("toy: click rule, row/col swapped", swapped, lambda r: r["n_applies"] < 0.2 * nc)):
            r = check_rule(src, "toy", td)
            good = r["ok"] and want(r)
            ok_all &= good
            print(f"   [{'PASS' if good else 'FAIL'}] {name:<32} applies to {r.get('n_applies')}, "
                  f"exactly right on {r.get('accuracy', 0):.0%}")
    d9 = RuleData("ft09", d)
    right = {}
    for name, src in (("as recorded", FT09_FLIP_RULE),
                      ("swapped", FT09_FLIP_RULE.replace("y, x = act.click", "x, y = act.click"))):
        r = check_rule(src, "ft09", d, split="train")
        moved = np.array([not np.array_equal(d9.before[i][d9.live], d9.after[i][d9.live])
                          for i in r["idx"]])
        right[name] = r["correct"] & moved
    # a click near the diagonal lands in the same tile read either way, so count
    # the tile changes that only ONE reading of the click predicts
    ours = int((right["as recorded"] & ~right["swapped"]).sum())
    theirs = int((right["swapped"] & ~right["as recorded"]).sum())
    good = ours >= 100 and theirs <= ours / 10
    ok_all &= good
    print(f"   [{'PASS' if good else 'FAIL'}] ft09 clicks decode as (row, col): {ours} real tile "
          f"changes are predicted only when the click is read that way, {theirs} only when "
          f"it is read swapped")

    print("\n3. The sandbox rejects bad rules at the right stage:")
    for name, (src, want) in BAD_RULES.items():
        r = check_rule(src, "ft09", d, max_moves=200)
        good = r["stage"] == want
        ok_all &= good
        print(f"   [{'PASS' if good else 'FAIL'}] {name:<18} -> {r['stage']}: {r['reason'][:70]}")
    print(f"\n[selftest] {'all passed' if ok_all else 'FAILED'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "selftest"])
    ap.add_argument("--dir", default=DATA_DIR)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.results, a.dir)
    else:
        raise SystemExit(0 if selftest(a.dir) else 1)


if __name__ == "__main__":
    main()
