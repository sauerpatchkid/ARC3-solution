"""heur_referee.py — the referee that grades heuristics against the recordings.

LABELS. For every recorded level completion, S* is the board one move before the
level was solved. Each earlier move in that level is labelled by hindsight:
"toward" if it left fewer non-ticker cells different from S* than there were
before it, "away" if it left more (unchanged moves are dropped). This is Probe
A's post-hoc signal (probe_hindsight.py) turned into a frozen, reusable dataset.
The move(s) whose result IS S* are excluded: they are toward by construction.

GRADING. For each labelled move, the heuristic scores every legal action on the
board the move was made from. The move actually played gets its PERCENTILE among
them (ties count half, so a heuristic with no opinion scores exactly 0.5). The
grade is the AUC: how often a "toward" move gets a higher percentile than an
"away" move. 0.5 = no information, 1.0 = perfect. The 95% CI bootstraps over
level completions, because moves from one completion are not independent.

"Legal actions" for a game are the action types Goose actually used there: the
buttons it pressed and, if it ever clicked, all 4,096 click positions.

Nothing here is ever shown to a heuristic except board, level layout, ticker and
recent history (heur_api.HeuristicAPI). S* is stored for the self-test's oracle
control only.

    uv run python -m llm_track.heur_referee build      # -> results/llm/referee/v1/
    uv run python -m llm_track.heur_referee selftest   # oracle, baseline, sandbox checks
"""
import argparse
import collections
import json
import os
import subprocess
import time

import numpy as np

from .corpus import (GOOSE, CorpusReader, find_corpora, game_of,
                     is_post_fix_boundary, iter_shards, level_events)
from .heur_api import API_DOC, HISTORY, HeuristicAPI, Move, background_of, regions
from .probe_pairs import build_masks, segment_start
from .serializer import serialize

DEFAULT_DIR = "results/llm/referee/v2"     # v2: rebuilt after the 2026-09-10 harvest
SAMPLES_PER_CLASS = 300     # per completion: up to 300 "toward" + 300 "away" moves


# ---------------------------------------------------------------- statistics
def rankdata(x):
    """1-based ranks; tied values share their mean rank."""
    _, inv, counts = np.unique(np.asarray(x, dtype=float), return_inverse=True,
                               return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2.0)[inv]


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(pos == neg): the Mann-Whitney AUC."""
    n1, n2 = len(pos), len(neg)
    if not n1 or not n2:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


# ---------------------------------------------------------------- building
def _label_range(r, a, b, s_star, live):
    """Hindsight labels (+1 toward S*, -1 away, 0 neither) for moves a..b."""
    out = np.empty(b - a + 1, dtype=np.int8)
    s0 = int(np.searchsorted(r.starts, a, side="right")) - 1
    s1 = int(np.searchsorted(r.starts, b, side="right")) - 1
    for s in range(s0, s1 + 1):
        lo, hi = max(a, int(r.starts[s])), min(b, int(r.starts[s + 1]) - 1)
        sh, base = r.shard(s), int(r.starts[s])
        fr = sh["frames"][lo - base:hi - base + 1]
        nf = sh["next_frames"][lo - base:hi - base + 1]
        before = ((fr != s_star) & live).sum(axis=(1, 2)).astype(np.int64)
        after = ((nf != s_star) & live).sum(axis=(1, 2)).astype(np.int64)
        out[lo - a:hi - a + 1] = np.sign(before - after)
    return out


def _history(r, j, seg0, mask, game, cache):
    """Up to HISTORY earlier moves in the same level, oldest first, stopping at a
    gap in action_nums (a reset or game over) - as arrays padded at the front."""
    an = r.scalars["action_nums"]
    ks, k = [], j - 1
    while k >= seg0 and len(ks) < HISTORY and an[k + 1] - an[k] == 1:
        ks.append(k)
        k -= 1
    act = np.zeros(HISTORY, np.int8)
    click = np.full((HISTORY, 2), -1, np.int8)
    changed = np.zeros(HISTORY, np.int16)
    move = np.full((HISTORY, 3), -1, np.int8)
    for pos, k in zip(range(HISTORY - len(ks), HISTORY), reversed(ks)):
        if k not in cache:
            t = r.get(k)
            rec = serialize(t["frame"], t["action"], t["next_frame"], mask, game=game)
            mv = max(rec["moves"], key=lambda q: q["size"]) if rec["moves"] else None
            cache[k] = (rec["action_type"], rec["click"], rec["n_changed_nonticker"], mv)
        a_type, clk, ch, mv = cache[k]
        act[pos] = a_type
        if clk:
            click[pos] = clk
        changed[pos] = min(ch, 32767)
        if mv:
            move[pos] = (mv["color"], mv["dy"], mv["dx"])
    return act, click, changed, move


def build(results="results", out=DEFAULT_DIR, per_class=SAMPLES_PER_CLASS, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    corpora = find_corpora(results, agent=GOOSE)
    with_events = []
    for name, d in corpora:
        r = CorpusReader(d)
        evs = level_events(r.scalars["levels"], r.scalars["action_nums"])
        if evs:
            with_events.append((name, d, r, evs))
    games = sorted({game_of(n) for n, _, _, _ in with_events})
    print(f"[referee] {len(corpora)} Goose corpora, {sum(len(e) for *_, e in with_events)} "
          f"level completions in {len(with_events)} of them, {len(games)} games")
    masks, mask_info, _ = build_masks([c for c in corpora if game_of(c[0]) in games])

    space = {g: {"buttons": set(), "clicks": False} for g in games}
    for name, d in corpora:
        g = game_of(name)
        if g in space:
            for s in iter_shards(d, fields=("actions",)):
                a = s["actions"]
                space[g]["buttons"] |= {int(x) for x in np.unique(a[a < 5])}
                space[g]["clicks"] |= bool((a >= 5).any())

    S = collections.defaultdict(list)
    events, firsts, stars, skipped = [], [], [], []
    for name, d, r, evs in with_events:
        g, sc = game_of(name), r.scalars
        mask = masks[g]
        run_id = os.path.basename(os.path.dirname(os.path.dirname(d)))
        for i, an, lo, hi in evs:
            post = is_post_fix_boundary(sc["action_nums"], i)
            t_i = r.get(i)
            s_star = t_i["frame"] if post else t_i["next_frame"]
            end = i - 2 if post else i - 1          # drop the move(s) whose result is S*
            seg0 = segment_start(sc["levels"], i)
            ev_name = f"{run_id}/{name}/{an}"
            if end - seg0 < 1:
                skipped.append((ev_name, "level too short")); continue
            lab = _label_range(r, seg0, end, s_star, ~mask)
            tw, aw = np.flatnonzero(lab > 0) + seg0, np.flatnonzero(lab < 0) + seg0
            if not len(tw) or not len(aw):
                skipped.append((ev_name, "no toward or no away moves")); continue
            pick = np.sort(np.concatenate([
                rng.choice(tw, min(per_class, len(tw)), replace=False),
                rng.choice(aw, min(per_class, len(aw)), replace=False)]))
            e = len(events)
            events.append({"game": g, "level": int(lo), "event": ev_name, "corpus": d,
                           "post_fix": bool(post), "level_moves": int(end - seg0 + 1),
                           "toward": int(len(tw)), "away": int(len(aw)),
                           "sampled": int(len(pick))})
            firsts.append(r.get(seg0)["frame"])
            stars.append(s_star)
            cache = {}
            for j in pick:
                t = r.get(int(j))
                S["boards"].append(t["frame"]); S["actions"].append(t["action"])
                S["labels"].append(1 if lab[j - seg0] > 0 else -1)
                S["event"].append(e); S["moves_in_level"].append(int(j - seg0))
                ha, hc, hch, hm = _history(r, int(j), seg0, mask, g, cache)
                S["hist_action"].append(ha); S["hist_click"].append(hc)
                S["hist_changed"].append(hch); S["hist_move"].append(hm)
        print(f"  {name:<14} {len(evs)} completion(s)  [{time.time() - t0:.0f}s]")

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "samples.npz"),
                        boards=np.stack(S["boards"]).astype(np.uint8),
                        actions=np.array(S["actions"], np.int32),
                        labels=np.array(S["labels"], np.int8),
                        event=np.array(S["event"], np.int32),
                        moves_in_level=np.array(S["moves_in_level"], np.int32),
                        hist_action=np.stack(S["hist_action"]),
                        hist_click=np.stack(S["hist_click"]),
                        hist_changed=np.stack(S["hist_changed"]),
                        hist_move=np.stack(S["hist_move"]))
    np.savez_compressed(os.path.join(out, "events.npz"),
                        first=np.stack(firsts).astype(np.uint8),
                        s_star=np.stack(stars).astype(np.uint8))
    np.savez_compressed(os.path.join(out, "masks.npz"), **{g: masks[g] for g in games})
    with open(os.path.join(out, "events.json"), "w") as f:
        json.dump(events, f, indent=1)
    per = collections.Counter((e["game"], e["level"]) for e in events)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit,
            "samples_per_class": per_class, "seed": seed, "history": HISTORY,
            "n_samples": len(S["labels"]), "n_events": len(events),
            "completions": {f"{g} L{lv}->L{lv + 1}": n for (g, lv), n in sorted(per.items())},
            "action_space": {g: {"buttons": sorted(v["buttons"]), "clicks": v["clicks"]}
                             for g, v in space.items()},
            "masks": mask_info, "skipped": skipped}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[referee] {len(events)} completions, {len(S['labels'])} labelled moves -> {out} "
          f"({time.time() - t0:.0f}s)")
    return meta


# ---------------------------------------------------------------- the referee
def _readonly(a):
    a = np.asarray(a)
    a.flags.writeable = False
    return a


class Referee:
    """Loads a built dataset; builds heuristic inputs; grades percentiles."""

    def __init__(self, d=DEFAULT_DIR, light=False):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.events = json.load(open(os.path.join(d, "events.json")))
        z = np.load(os.path.join(d, "samples.npz"))
        self.labels, self.event, self.actions = z["labels"], z["event"], z["actions"]
        self.ev_game = np.array([e["game"] for e in self.events])
        self.ev_level = np.array([e["level"] for e in self.events])
        if light:                               # grading needs only labels and events
            return
        self.boards = z["boards"]
        self.moves_in_level = z["moves_in_level"]
        self.h = (z["hist_action"], z["hist_click"], z["hist_changed"], z["hist_move"])
        ev = np.load(os.path.join(d, "events.npz"))
        self.first, self.s_star = ev["first"], ev["s_star"]
        mk = np.load(os.path.join(d, "masks.npz"))
        self.masks = {g: _readonly(mk[g]) for g in mk.files}
        self._layout = {}
        self._space = {}

    def select(self, game=None, level=None, exclude_game=None):
        """Sample indices whose completion matches."""
        keep = np.ones(len(self.events), bool)
        if game is not None:
            keep &= np.isin(self.ev_game, [game] if isinstance(game, str) else list(game))
        if exclude_game is not None:
            keep &= self.ev_game != exclude_game
        if level is not None:
            keep &= self.ev_level == level
        return np.flatnonzero(keep[self.event])

    def inputs(self, i):
        """(board, api) exactly as the live agent would build them. Read-only."""
        e = int(self.event[i])
        g = self.events[e]["game"]
        tick = self.masks[g]
        if e not in self._layout:
            first = _readonly(self.first[e].copy())
            lay = regions(first, background_of(first), tick)
            for reg in lay:
                reg.cells.flags.writeable = False
                reg.mask.flags.writeable = False
            self._layout[e] = (first, lay)
        first, lay = self._layout[e]
        act, clk, ch, mv = (h[i] for h in self.h)
        hist = [Move(int(act[k]), (int(clk[k, 0]), int(clk[k, 1])) if clk[k, 0] >= 0 else None,
                     int(ch[k]), (int(mv[k, 0]), int(mv[k, 1]), int(mv[k, 2])) if mv[k, 0] >= 0 else None)
                for k in range(len(act)) if act[k] > 0]
        api = HeuristicAPI(first, tick, layout=lay, history=hist,
                           level=int(self.events[e]["level"]),
                           moves_in_level=int(self.moves_in_level[i]))
        return _readonly(self.boards[i].copy()), api

    def space(self, game):
        """(button indices, clicks allowed) for a game."""
        if game not in self._space:
            s = self.meta["action_space"][game]
            self._space[game] = (np.array(s["buttons"], dtype=int), bool(s["clicks"]))
        return self._space[game]

    def taken_percentile(self, i, buttons, clicks):
        """Percentile of the played action among the legal ones, and whether the
        heuristic scored every legal action the same."""
        btn, has_clicks = self.space(self.events[int(self.event[i])]["game"])
        vals = buttons[btn]
        if has_clicks:
            vals = np.concatenate([vals, clicks.ravel()])
        a = int(self.actions[i])
        v = buttons[a] if a < 5 else clicks.ravel()[a - 5]
        lo, hi = vals.min(), vals.max()
        if lo == hi:
            return 0.5, True
        less = np.count_nonzero(vals < v)
        equal = np.count_nonzero(vals == v)
        return (less + 0.5 * (equal - 1)) / (len(vals) - 1), False

    def grade(self, idx, pct, n_boot=1000, seed=0):
        """AUC of percentiles, toward vs away, with a CI bootstrapped over completions."""
        idx, pct = np.asarray(idx), np.asarray(pct, dtype=float)
        lab, ev = self.labels[idx], self.event[idx]
        a = auc(pct[lab > 0], pct[lab < 0])
        groups = {e: np.flatnonzero(ev == e) for e in np.unique(ev)}
        keys = list(groups)
        rng = np.random.default_rng(seed)
        stats = []
        for _ in range(n_boot if len(keys) > 1 else 0):
            sel = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
            p, l = pct[sel], lab[sel]
            if (l > 0).any() and (l < 0).any():
                stats.append(auc(p[l > 0], p[l < 0]))
        lo, hi = (np.percentile(stats, [2.5, 97.5]) if stats else (float("nan"),) * 2)
        return {"auc": round(a, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
                "n": int(len(idx)), "n_events": len(keys)}


# ---------------------------------------------------------------- self-test
# One-line rules a heuristic must beat. "objects" is shown for completeness but
# is no bar on ft09: every labelled ft09 move is already a click on an object,
# so it scores exactly 0.5 there. The two "touched" rules are the real bar.
SIMPLE_RULES = {
    "objects": '''IDEA = "prefer clicking on anything that is not background"
def score(board, api):
    buttons, clicks = api.zeros()
    clicks[(board != api.background) & ~api.ticker] = 1.0
    return buttons, clicks
''',
    "untouched": '''IDEA = "prefer clicking cells that still have their level-start colour"
def score(board, api):
    buttons, clicks = api.zeros()
    clicks[(board == api.first) & (board != api.background) & ~api.ticker] = 1.0
    return buttons, clicks
''',
    "touched": '''IDEA = "prefer clicking cells that changed since the level started"
def score(board, api):
    buttons, clicks = api.zeros()
    clicks[(board != api.first) & ~api.ticker] = 1.0
    return buttons, clicks
''',
}


def simple_rule_scores(d, where):
    """Grade every SIMPLE_RULES entry on a split; returns {name: evaluate-result}."""
    from .heur_sandbox import evaluate
    return {name: evaluate(src, d, where=where) for name, src in SIMPLE_RULES.items()}

BAD = {   # snippet -> the stage that must reject it
    "imports a module": ("import os\ndef score(board, api):\n    return api.zeros()", "static"),
    "opens a file": ("def score(board, api):\n    open('x')\n    return api.zeros()", "static"),
    "dunder escape": ("def score(board, api):\n    x = ().__class__\n    return api.zeros()", "static"),
    "numpy file I/O": ("def score(board, api):\n    np.load('x')\n    return api.zeros()", "static"),
    "wrong output shape": ("def score(board, api):\n    return np.zeros(5), np.zeros(10)", "runtime"),
    "writes to the board": ("def score(board, api):\n    board[0, 0] = 1\n    return api.zeros()", "runtime"),
    "no opinion at all": ("def score(board, api):\n    return api.zeros()", "gate"),
    "infinite loop": ("def score(board, api):\n    while True:\n        pass", "timeout"),
}


def selftest(d=DEFAULT_DIR):
    from .heur_sandbox import evaluate
    ref = Referee(d)
    print(f"[selftest] dataset {d}: {ref.meta['n_events']} completions, "
          f"{ref.meta['n_samples']} labelled moves")
    splits = [("ft09 level 1", dict(game="ft09", level=0)),
              ("ft09 level 2", dict(game="ft09", level=1)),
              ("other click games", dict(game=[g for g, s in ref.meta["action_space"].items()
                                               if s["clicks"] and g != "ft09"]))]

    print("\n1. Oracle control - NOT a heuristic: it peeks at S*, the solved board, and\n"
          "   prefers clicks on cells that differ from it. On ft09, where a click changes\n"
          "   the clicked tile, it must score near 1.0 or the labels / action indexing /\n"
          "   ranking are broken. It is NOT a valid check on the other click games: there\n"
          "   the clicked cell itself rarely changes (you click a control, something else\n"
          "   moves), so 'click where the board is wrong' is simply the wrong oracle.")
    for name, where in splits:
        idx = ref.select(**where)
        if not len(idx):
            print(f"   {name:<18} (no data)"); continue
        pct = np.empty(len(idx))
        for k, i in enumerate(idx):
            board, api = ref.inputs(i)
            e = int(ref.event[i])
            clicks = ((board != ref.s_star[e]) & ~api.ticker).astype(float)
            pct[k], _ = ref.taken_percentile(i, np.zeros(5), clicks)
        g = ref.grade(idx, pct)
        print(f"   {name:<18} AUC {g['auc']:.3f}  CI {g['ci'][0]:.3f}-{g['ci'][1]:.3f}  "
              f"({g['n']} moves, {g['n_events']} completions)")

    print("\n2. Simple rules, run end-to-end through the sandbox. The best of these on a\n"
          "   split is the bar a heuristic must beat there (uniform = 0.5 by construction):")
    for name, where in splits[:2]:
        for rule, r in simple_rule_scores(d, where).items():
            label = f"{rule}, {name}"
            if r["ok"]:
                print(f"   {label:<26} AUC {r['auc']:.3f}  CI {r['ci'][0]:.3f}-"
                      f"{r['ci'][1]:.3f}  ({r['ms_mean']} ms/call)")
            else:
                print(f"   {label:<26} {r['stage']}: {r['reason'][:60]}")

    print("\n3. Sandbox must reject bad code at the right stage:")
    allok = True
    for name, (src, want) in BAD.items():
        r = evaluate(src, d, where=dict(game="ft09", level=0), max_samples=200,
                     timeout_s=20 if want == "timeout" else 120)
        good = r["stage"] == want
        allok &= good
        print(f"   [{'PASS' if good else 'FAIL'}] {name:<20} -> {r['stage']}: {r['reason'][:70]}")
    # Regression checks for fixes made after the first writer smoke runs.
    r = evaluate("def score(board, api):\n    buttons, clicks = api.zeros()\n"
                 "    buttons[5] = 1.0\n    return buttons, clicks",
                 d, where=dict(game="ft09", level=0), max_samples=100)
    good = r["stage"] == "runtime" and r["reason"].startswith("IndexError")
    allok &= good
    print(f"   [{'PASS' if good else 'FAIL'}] {'real error surfaces':<20} -> {r['stage']}: {r['reason'][:70]}")
    example = "\n".join(line[4:] for line in
                        API_DOC.split("not the answer for this game:\n\n")[1].splitlines())
    r = evaluate(example, d, where=dict(game="ft09", level=0), max_samples=500)
    good = r["stage"] == "graded"
    allok &= good
    print(f"   [{'PASS' if good else 'FAIL'}] {'API_DOC example runs':<20} -> {r['stage']}: "
          f"{r.get('ms_mean')} ms/call")
    print(f"\n[selftest] sandbox checks {'all passed' if allok else 'FAILED'}")
    return allok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "selftest"])
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.results, a.dir)
    else:
        raise SystemExit(0 if selftest(a.dir) else 1)


if __name__ == "__main__":
    main()
