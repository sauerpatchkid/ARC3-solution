"""s5_act.py - S5 ACT: the rule book steers Goose online, and the same code replays a recording.

The problem it fixes: no rule has ever been used online. Goose still spends moves on
actions the book knows do nothing, and its map walks across a level to press a button
the book knows does nothing (plan section 5.5).

Version 1 builds PRUNE, goal HINTS and the map's part (GbMap). PROBE, IMAGINE and
contact goals come later (the plan's cut line keeps prune and hints first).

  PRUNE  A move the carried no-ops cover gets Goose's probability x `alpha` (never
         zero, so a contradiction can still show up; the first one drops whatever
         covered the move, logged). Two kinds of no-op are carried:
          - COLOUR no-ops (no LLM; Rulebook v2's strict C2-free rule): a colour
            clicked at least `noop_min_clicks` times on one level with nothing ever
            changing is a no-op on every later level, until a click on it changes
            something.
          - the book's no-op RULES: per action group, the candidate with the most
            moves among those meeting v1's known-no-op bar on the training level
            (it applies to >= 20 moves, predicts no change, and the whole board
            indeed stayed as it was; gb/core/check.grade). On every level, the
            training level too, a rule waits (s4_carry.ready), is re-bound on the
            level's own moves so far (bind.search, its training binding preferred,
            S0's mask), and is used only if the binding is exact there, applies to
            at least `noop_rule_min_cases` cases, and the whole board stayed
            unchanged on every one of them. If not, it is tried again when the
            level has twice the moves, until 4 x `online_max_wait` moves. (Online the decoration mask, and so a
            Region slot's numbering, is not known when a level starts, so even the
            training binding is checked on the level's own moves first.)
  HINTS  At each level-up, the 2 colours that vanished or shrank most between the
         level's first board and the board before the winning move (Rulebook v2's
         design, after BDR-Pro). On the next level a click on a hint colour gets
         x (1 + `gamma`); a hint is dropped after `hint_drop_moves` moves in which
         no cell of that colour changed.
  MAP    GbMap (a ReturnMap subclass; return_map.py is never edited): on levels after
         the training level (where a rule has then been exact on two levels: the
         plan's axiom), a button that a re-bound no-op rule covers on a screen
         counts as already tried there, so the map never walks across a level to
         press it.

How a weight reaches Goose. Goose samples a move from its own distribution, as
always, so its random stream is untouched. The guide accepts the move with
probability w(move) / w_max and otherwise draws again from the same distribution
with its own generator. That is rejection sampling: the move taken has probability
exactly p(move) w(move) / sum(p w) (up to a cap of MAX_DRAWS draws), and the guide
only judges the moves it is shown, not all 4,101.

Rules run in one persistent child process (RuleWorker), the sandbox's restricted
namespace, with a timeout per question: a rule that hangs or crashes switches the
rules off for the rest of the run and never stops Goose.

REPLAY (the offline counterfactual, plan 6.2). `replay()` feeds one recorded run
through the same Guide, level by level, and asks of every recorded move whether it
was covered and whether it changed anything. In the first `waste_window` actions of
each level from level 2 up: AVOIDABLE WASTE = covered moves that indeed changed
nothing, MASKING RISK = covered moves that changed something (Rulebook v2's
definitions; counted in recorded moves, reported as shares of the window's).
"Changed" means a live cell changed: in the replay under the level's decoration mask
built as S0 builds it; online under the agent's own (the map's) mask.

Reads  (online) results/glassbox/<book run>/s2_write/<game>/candidates.jsonl and
       s3_check/<game>_L<k>.book.json (EVAL_GB_BOOK; none = the GB-free arm)
Writes (online) <run>/<game>/glassbox/events.jsonl, s5_stats.json
       (replay) whatever the caller writes from replay()'s result
"""
import collections
import glob
import json
import multiprocessing as mp
import os
import tempfile

import numpy as np
import xxhash

from gb.core import bind as B
from gb.core.evidence import GRID, LevelEvidence, read_moves, ticker_mask
from gb.s0_mask import complete_bars
from gb.s4_carry import ready
from return_map import ReturnMap
from upgrades import UpgradedReturnMap

VERSION = 1
DEFAULTS = {"prune": True, "hints": True, "alpha": 0.1, "gamma": 1.0, "hint_drop_moves": 2000,
            "noop_min_clicks": 20, "noop_rule_min_cases": 10, "online_min_cases": 10,
            "online_max_wait": 1000, "waste_window": 2000, "n_hints": 2, "ask_timeout_s": 5.0}
MAX_DRAWS = 100                  # rejection sampling: the move drawn last is taken
MAX_CACHE = 200_000              # (screen, move) answers kept before the cache starts over
CLICKS = 5                       # unified action index of the first click (5 + 64*y + x)


def family(action):
    """bind.family_of for one unified action index: its button, or 'click'."""
    return f"ACTION{action + 1}" if action < CLICKS else "click"


def clicked_colour(board, action):
    y, x = divmod(int(action) - CLICKS, GRID)
    return int(board[y, x])


# ---- the book's no-op rules ------------------------------------------------------------------------
def noop_rules(book_dir, game):
    """Per action group, the known no-op with the most moves (check.pick_book's choice),
    from a finished offline run: [{id, group, code, binding}], and the training level
    (1-based, as the evidence files are named). ([], None) if the run has no book for it."""
    books = glob.glob(os.path.join(book_dir, "s3_check", f"{game}_L*.book.json"))
    p_cands = os.path.join(book_dir, "s2_write", game, "candidates.jsonl")
    if not books or not os.path.exists(p_cands):
        return [], None
    with open(books[0]) as f:
        level = json.load(f)["level"]
    best = {}
    with open(p_cands) as f:
        for c in map(json.loads, f):
            g = c.get("grade") or {}
            if c.get("stage") == "checked" and g.get("known_noop") and not c.get("duplicate"):
                old = best.get(c["group"])
                if old is None or (g["moves"], -c["order"]) > (old["grade"]["moves"], -old["order"]):
                    best[c["group"]] = c
    rules = [{"id": c["id"], "group": c["group"], "code": c["code"], "binding": c.get("binding") or {}}
             for c in sorted(best.values(), key=lambda c: c["order"])]
    return rules, level


# ---- rules in a child process ------------------------------------------------------------------------
def _worker(conn):
    """Child process: holds the level's bound rules and answers which of them cover a move."""
    from gb.core import contract as C
    from gb.core.evidence import Act, Api
    rules, api = [], None
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        try:
            if msg[0] == "level":                       # ("level", first, mask, level0, [(id, code, binding)])
                _, first, mask, level0, specs = msg
                api = Api(first, mask, level0)
                rules = []
                for rid, code, binding in specs:
                    r = C.Rule(code)
                    rules.append((rid, r, r.values(binding, api.layout)))
                conn.send(("ok", len(rules)))
            elif msg[0] == "ask":                       # ("ask", board, action, ids) -> ids that cover it
                _, board, action, ids = msg
                board.flags.writeable = False
                live = ~api.ticker
                out = []
                for rid, r, p in rules:
                    if rid not in ids:
                        continue
                    act = Act(action)
                    if r.applies(board, act, api, p):
                        pred, claim = r.predict(board, act, api, p, live)
                        if not (np.asarray(pred) != board)[claim].any():
                            out.append(rid)
                conn.send(("ok", out))
        except Exception as e:                          # a rule's own error: it covers nothing
            conn.send(("error", f"{type(e).__name__}: {e}"))


class RuleWorker:
    """The book's rules in one spawned child (the sandbox's restricted namespace), so a
    rule that hangs or crashes cannot stop Goose: a question that times out kills the
    child and switches the rules off for the rest of the run."""

    def __init__(self, timeout_s=DEFAULTS["ask_timeout_s"]):
        self.timeout_s, self.proc, self.conn, self.dead = timeout_s, None, None, None

    def _call(self, msg):
        if self.dead:
            return None
        if self.proc is None:
            ctx = mp.get_context("spawn")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_worker, args=(child,), daemon=True)
            self.proc.start()
        self.conn.send(msg)
        if not self.conn.poll(self.timeout_s if msg[0] == "ask" else 60.0):
            self.stop(f"no answer within {self.timeout_s}s")
            return None
        kind, val = self.conn.recv()
        return val if kind == "ok" else None

    def level(self, first, mask, level0, specs):
        return self._call(("level", np.asarray(first, np.uint8), np.asarray(mask, bool), level0, specs))

    def ask(self, board, action, ids):
        got = self._call(("ask", np.asarray(board, np.uint8), int(action), set(ids)))
        return got or []

    def stop(self, why=None):
        if self.proc is not None:
            try:
                self.conn.send(("stop",))
            except (OSError, BrokenPipeError):
                pass
            self.proc.join(2)
            if self.proc.is_alive():
                self.proc.terminate()
        self.proc = None
        if why:
            self.dead = why


# ---- the guide ---------------------------------------------------------------------------------------
class Guide:
    """S5's state for one game: carried no-ops and goal hints, level by level (module doc).
    The caller reports each level's start (`new_level`) and each move (`observe`), and
    asks `override` (online) or `covered` (replay)."""

    def __init__(self, game, rules=(), train_level=None, settings=None, seed=0, log_path=None, worker=None):
        self.game = game
        self.s = dict(DEFAULTS, **(settings or {}))
        self.rules = {r["id"]: dict(r) for r in rules}
        self.train_level = train_level
        self.rng = np.random.default_rng(seed)
        self.worker = worker if worker is not None else (RuleWorker(self.s["ask_timeout_s"]) if rules else None)
        self.log_path = log_path
        self.level = None                       # 1-based, as the evidence files are named
        self.colour_noops = {}                  # colour -> level it was carried from
        self.hints = {}                         # colour -> moves since a cell of it changed
        self.first = self.mask = None           # the level's first board; S0's mask (hints, fallback)
        self.rule_mask = None                   # the mask the rules were re-bound with on this level
        self.clicks = collections.Counter()     # colour -> clicks this level
        self.click_changed = collections.Counter()
        self.moves = []                         # this level's moves, kept until every rule is re-bound
        self.changing = collections.defaultdict(set)   # family -> hashes of distinct changing moves
        self.active = {}                        # rule id -> binding in use on this level
        self.pending = set()                    # rule ids waiting to be re-bound
        self.not_before = {}                    # rule id -> moves on the level before it is tried again
        self.cache = {}                         # (screen hash, action) -> what covers it, or None
        self.moves_on_level = 0
        self.rules_version = 0                  # bumped whenever the rules in use change (GbMap's cache)
        self.stats = collections.Counter()
        self.events = []

    # -- events ---------------------------------------------------------------------------------
    def _event(self, kind, **kw):
        e = dict(kind=kind, level=self.level, move=self.moves_on_level, **kw)
        self.events.append(e)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(e) + "\n")

    def new_level(self, level, first, mask=None, before_win=None):
        """A level starts (1-based `level`; `first` its first board; `mask` a decoration
        mask for goal hints and for `observe` when it is not told whether a move changed
        anything; `before_win` the board before the previous level's winning move)."""
        if self.level is not None:
            for c in sorted(self.clicks):          # carry colours that never did anything
                if self.clicks[c] >= self.s["noop_min_clicks"] and not self.click_changed[c] \
                        and c not in self.colour_noops:
                    self.colour_noops[c] = self.level
                    self._event("colour no-op carried", colour=c, clicks=self.clicks[c])
            if self.s["hints"] and before_win is not None and self.first is not None:
                self.hints = {c: 0 for c in goal_colours(self.first, before_win, self.mask, self.s["n_hints"])}
                if self.hints:
                    self._event("goal hints", colours=sorted(self.hints))
        self.level, self.first = level, np.array(first, np.uint8)
        self.mask = np.zeros((GRID, GRID), bool) if mask is None else np.asarray(mask, bool)
        self.clicks, self.click_changed = collections.Counter(), collections.Counter()
        self.moves, self.changing, self.cache, self.moves_on_level = [], collections.defaultdict(set), {}, 0
        self.active, self.pending, self.rule_mask, self.not_before = {}, set(self.rules), None, {}

    def _bind_worker(self):
        self.rules_version += 1
        self.cache = {}
        specs = [(i, self.rules[i]["code"], b) for i, b in self.active.items()]
        if self.worker is not None and specs and \
                self.worker.level(self.first, self.rule_mask, self.level - 1, specs) is None:
            self._event("rules off", why=self.worker.dead or "the rules could not be loaded")
            self.active = {}

    def observe(self, before, action, after, changed=None):
        """One move made on the current level (unified action index)."""
        before, after = np.asarray(before), np.asarray(after)
        if changed is None:
            changed = bool(((before != after) & ~self.mask).any())
        self.moves_on_level += 1
        who = self.covered(before, action) if changed else ()
        if who:                                     # a contradiction: drop whatever covered it
            for w in who:
                if w.startswith("colour "):
                    c = int(w.split()[1])
                    if self.colour_noops.pop(c, None) is not None:
                        self._event("colour no-op dropped", colour=c)
                else:
                    rid = int(w.split()[1])
                    if self.active.pop(rid, None) is not None:
                        self._event("rule demoted", rule=rid)
                        self._bind_worker()
            self.cache = {}
            self.stats["contradictions"] += 1
        if action >= CLICKS:
            c = clicked_colour(before, action)
            self.clicks[c] += 1
            self.click_changed[c] += int(changed)
        if self.hints:
            diff = (before != after) & ~self.mask
            for c in list(self.hints):
                self.hints[c] = 0 if (diff & ((before == c) | (after == c))).any() else self.hints[c] + 1
                if self.hints[c] >= self.s["hint_drop_moves"]:
                    del self.hints[c]
                    self._event("goal hint dropped", colour=c)
        if self.pending:
            self.moves.append((before.astype(np.uint8), int(action), after.astype(np.uint8)))
            if changed:
                self.changing[family(action)].add((screen_key(before), int(action)))
            for rid in sorted(self.pending):
                fam = B.family_of(self.rules[rid]["group"])
                if ready(self.moves_on_level, len(self.changing[fam]), self.s) \
                        and self.moves_on_level >= self.not_before.get(rid, 0):
                    self.pending.discard(rid)
                    self._rebind(rid)
            if not self.pending:
                self.moves = []

    def _rebind(self, rid):
        """Re-bind one no-op rule on the level's moves so far (module doc, PRUNE)."""
        r = self.rules[rid]
        f, a, n = (np.array(x) for x in zip(*self.moves))
        mask = self.rule_mask if self.rule_mask is not None else level_mask(f, n)
        ev = LevelEvidence.from_moves(self.game, self.level - 1, f, n, a.astype(np.int32), mask=mask)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "level.npz")
            ev.save(p)
            found = B.search(r["code"], p, family=B.family_of(r["group"]), prefer=r["binding"])
        best, flag = B.choose(found) if found.get("stage") == "checked" else (None, "binding failed")
        res = best["res"] if best else None
        ok = (res is not None and int(res["applies"].sum()) >= self.s["noop_rule_min_cases"]
              and bool(res["correct"][res["applies"]].all()) and not bool(res["changes"][res["applies"]].any()))
        self._event("rule re-bound" if ok else "rule not re-bound", rule=rid, flag=flag,
                    binding=best["binding"] if best else None,
                    applies=int(res["applies"].sum()) if res is not None else 0, moves=len(self.moves))
        if ok:
            self.active[rid], self.rule_mask = best["binding"], mask
            self._bind_worker()
        elif 2 * self.moves_on_level <= 4 * self.s["online_max_wait"]:     # try again with more of the level
            self.pending.add(rid)
            self.not_before[rid] = 2 * self.moves_on_level

    # -- what covers a move ------------------------------------------------------------------------
    def covered(self, board, action):
        """What covers this move as a no-op ('colour c', 'rule i'), as a tuple; () if nothing."""
        if not self.s["prune"]:
            return ()
        key = (screen_key(board), int(action))
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= MAX_CACHE:
            self.cache = {}
        who = []
        if action >= CLICKS and clicked_colour(board, action) in self.colour_noops:
            who.append(f"colour {clicked_colour(board, action)}")
        ids = [i for i in self.active if B.family_of(self.rules[i]["group"]) == family(action)]
        if ids and self.worker is not None:
            got = self.worker.ask(board, action, ids)
            if self.worker.dead and self.active:
                self._event("rules off", why=self.worker.dead)
                self.active = {}
            who += [f"rule {i}" for i in got]
        self.cache[key] = tuple(who)
        return self.cache[key]

    def weight(self, board, action):
        w = self.s["alpha"] if self.covered(board, action) else 1.0
        if self.hints and action >= CLICKS and clicked_colour(board, action) in self.hints:
            w *= 1.0 + self.s["gamma"]
        return w

    def override(self, action_idx, coords, coord_idx, all_probs, board):
        """Online: Goose's sampled move, accepted with probability w / w_max, else drawn
        again from Goose's distribution with the guide's generator (module doc)."""
        a = action_idx if action_idx < CLICKS else CLICKS + coord_idx
        w_max = 1.0 + self.s["gamma"] if self.hints else 1.0
        p, first = None, a
        for _ in range(MAX_DRAWS):
            w = self.weight(board, a)
            self.stats["draws"] += 1
            if w >= w_max or self.rng.random() < w / w_max:
                break
            if p is None:
                p = np.asarray(all_probs, np.float64).copy()
                p[CLICKS:] /= GRID * GRID
                if not np.isfinite(p).all() or p.sum() <= 0:
                    break
                p /= p.sum()
            a = int(self.rng.choice(len(p), p=p))
        self.stats["moves"] += 1
        self.stats["covered_first"] += int(bool(self.covered(board, first)))
        self.stats["changed_move"] += int(a != first)
        return ReturnMap.unpack(a)

    # -- reporting ---------------------------------------------------------------------------------
    def summary(self):
        return dict(self.stats, colour_noops=sorted(self.colour_noops), hints=sorted(self.hints),
                    active_rules=sorted(self.active), rules=len(self.rules), train_level=self.train_level,
                    rules_off=getattr(self.worker, "dead", None))

    def close(self):
        if self.worker is not None:
            self.worker.stop()


def screen_key(board):
    return xxhash.xxh64(np.ascontiguousarray(board, np.uint8).tobytes()).intdigest()


def goal_colours(first, before_win, mask, n=2):
    """The `n` colours that vanished or shrank most from a level's first board to the
    board before its winning move (cells outside the decoration mask)."""
    live = ~np.asarray(mask, bool)
    a = np.bincount(np.asarray(first)[live].ravel(), minlength=16)
    b = np.bincount(np.asarray(before_win)[live].ravel(), minlength=16)
    drop = a - b
    order = sorted((c for c in range(16) if drop[c] > 0), key=lambda c: (b[c] > 0, -drop[c], c))
    return order[:n]


# ---- the map's part ----------------------------------------------------------------------------------
class GbMap(UpgradedReturnMap):
    """The return map, with buttons a re-bound no-op rule covers counted as tried
    (module doc, MAP). Same settings and upgrades as the map it replaces."""

    def __init__(self, guide, *args, **kw):
        super().__init__(*args, **kw)
        self.guide = guide
        self.noop_buttons = {}              # key -> buttons the guide's rules cover there
        self.noop_version = None            # the guide's rules_version they were worked out with
        self.stats["gb_noop_buttons"] = 0

    @classmethod
    def wrap(cls, guide, m):
        return cls(guide, stall=m.stall, click_tries=m.click_tries, max_route=m.max_route,
                   return_after_game_over=m.return_after_game_over, max_nodes=m.max_nodes, rng=m.rng,
                   gated=getattr(m, "gated", False), bars=getattr(m, "bars", None))

    def clear(self, new_level=True):
        super().clear(new_level)
        self.noop_buttons = {}

    def observe(self, raw, live_mask, has_prev, available_actions):
        k = super().observe(raw, live_mask, has_prev, available_actions)
        g = self.guide
        if self.noop_version != g.rules_version:
            self.noop_buttons, self.noop_version = {}, g.rules_version
        if k in self.edges and k not in self.noop_buttons and g.level is not None and g.train_level is not None \
                and g.level > g.train_level and g.active:
            self.noop_buttons[k] = frozenset(b for b in self.buttons.get(k, ())
                                             if any(w.startswith("rule ") for w in g.covered(raw, b)))
            self.stats["gb_noop_buttons"] += len(self.noop_buttons[k])
        return k

    def _untried_buttons(self, k):
        return super()._untried_buttons(k) - self.noop_buttons.get(k, frozenset())


# ---- online: what the agent talks to ---------------------------------------------------------------
class Online:
    """What custom_agents/action.py holds when EVAL_GB=1: the guide, plus the bookkeeping
    of a live run (a level starts when the score changes; the board before the winning
    move is the last one seen; "changed" is judged under the agent's own mask)."""

    def __init__(self, guide, book=None):
        self.guide, self.book = guide, book
        self.score, self.last_raw, self.map = None, None, None

    def wrap_map(self, m):
        """The agent's return map, as a GbMap with the same settings and upgrades."""
        self.map = GbMap.wrap(self.guide, m)
        return self.map

    def _mask(self, canon_mask):
        m = np.zeros((GRID, GRID), bool) if canon_mask is None else np.asarray(canon_mask, bool)
        bars = getattr(self.map, "bars", None)
        return m | bars.sticky if bars is not None else m

    def step(self, prev_raw, prev_action, cur_raw, score, canon_mask):
        """Each decision frame. `prev_raw` is None right after a reset or a level change."""
        mask = self._mask(canon_mask)
        if score != self.score:
            self.guide.new_level(int(score) + 1, cur_raw, mask,
                                 before_win=self.last_raw if self.score is not None else None)
            self.score = score
        elif prev_raw is not None and prev_action is not None:
            self.guide.observe(prev_raw, int(prev_action), cur_raw, bool(((prev_raw != cur_raw) & ~mask).any()))
        self.last_raw = np.array(cur_raw, np.uint8)

    def override(self, action_idx, coords, coord_idx, all_probs, board):
        return self.guide.override(action_idx, coords, coord_idx, all_probs, board)

    def config(self):
        """Fields for run_config.json."""
        return {"gb": True, "gb_book": self.book, "gb_rules": sorted(self.guide.rules),
                "gb_train_level": self.guide.train_level, "gb_version": VERSION,
                "gb_settings": {k: self.guide.s[k] for k in sorted(self.guide.s)}}

    def dump(self, path):
        """At exit: the guide's counts, and the rules' child process stopped."""
        try:
            with open(path, "w") as f:
                json.dump(dict(self.guide.summary(), **(self.map.summary() if self.map is not None else {})),
                          f, indent=1, default=int)
        except OSError:
            pass
        self.guide.close()


def from_env(game_id, seed, out_dir):
    """The online guide for one run: EVAL_GB_BOOK names an offline run under
    results/glassbox/ (or a path) whose book's no-op rules are carried; unset, the guide
    has no rules (the GB-free arm: colour no-ops and goal hints only)."""
    from gb import artifacts as A
    game = game_id.split("-")[0]
    book = os.getenv("EVAL_GB_BOOK", "").strip() or None
    rules, level = [], None
    if book:
        book_dir = book if os.path.isabs(book) else os.path.join(A.results_dir(), book)
        if not os.path.isdir(book_dir):
            raise SystemExit(f"EVAL_GB_BOOK: no such run {book_dir}")
        rules, level = noop_rules(book_dir, game)
    config = A.load_config()
    settings = dict(DEFAULTS, **{k: v for k, v in config.get("act", {}).items() if k in DEFAULTS},
                    **{k: v for k, v in config.get("carry", {}).items()
                       if k in ("noop_min_clicks", "online_min_cases", "online_max_wait")})
    os.makedirs(out_dir, exist_ok=True)
    g = Guide(game, rules, level, settings, seed=seed + 7919, log_path=os.path.join(out_dir, "events.jsonl"))
    g._event("start", book=book, rules=[r["id"] for r in rules], train_level=level,
             settings={k: settings[k] for k in sorted(settings)})
    return Online(g, book)


# ---- replay: the offline counterfactual --------------------------------------------------------------
def level_mask(frames, next_frames):
    """A level's decoration mask as S0 builds it: the ticker scan, bars run to the edges."""
    mask, _ = ticker_mask(frames[:20000], next_frames[:20000])
    return complete_bars(mask)


def replay(guide, corpus, max_level=None, log=print):
    """Feed one recorded run through `guide`, level by level (module doc, REPLAY).
    Returns per level {level, moves, window, covered, waste, masking, by}."""
    out, prev_last = [], None
    level0 = 0
    while max_level is None or level0 < max_level:
        f, n, a, order = read_moves(corpus, level0)
        if not len(a):
            break
        mask = level_mask(f, n)
        guide.new_level(level0 + 1, f[0], mask, before_win=prev_last)
        nums = order["num"] - order["num"][0] if order else np.arange(len(a))
        row = dict(level=level0 + 1, moves=int(len(a)), window=0, covered=0, waste=0, masking=0,
                   by=collections.Counter())
        for i in range(len(a)):
            changed = bool(((f[i] != n[i]) & ~mask).any())
            if nums[i] < guide.s["waste_window"]:
                who = guide.covered(f[i], int(a[i]))
                row["window"] += 1
                if who:
                    row["covered"] += 1
                    row["waste" if not changed else "masking"] += 1
                    for w in who:
                        row["by"][w.split()[0]] += 1
            guide.observe(f[i], int(a[i]), n[i], changed)
        row["by"] = dict(row["by"])
        out.append(row)
        log(f"[{guide.game}] level {level0 + 1}: {row['moves']} moves; first {guide.s['waste_window']} actions: "
            f"{row['window']} moves, covered {row['covered']}, waste {row['waste']}, masking {row['masking']}")
        prev_last = n[-1]
        level0 += 1
    return out


def replay_run(run, game, seed, corpus, book_dir=None, log=print):
    """One recorded run replayed into <run>/s5_act/<game>_s<seed>.replay.json: per level
    (module doc, REPLAY) and the guide's events. `book_dir` None: no rules (GB-free)."""
    rules, level = noop_rules(book_dir, game) if book_dir else ([], None)
    settings = dict(DEFAULTS, **{k: v for k, v in run.config.get("act", {}).items() if k in DEFAULTS},
                    **{k: v for k, v in run.config.get("carry", {}).items()
                       if k in ("noop_min_clicks", "online_min_cases", "online_max_wait")})
    guide = Guide(game, rules, level, settings, seed=seed + 7919)
    try:
        rows = replay(guide, corpus, log=log)
    finally:
        guide.close()
    out = os.path.join(run.stage_dir("s5_act"), f"{game}_s{seed}.replay.json")
    with open(out, "w") as f:
        json.dump({"game": game, "seed": seed, "corpus": corpus, "book": book_dir, "rules": [r["id"] for r in rules],
                   "train_level": level, "settings": settings, "levels": rows, "events": guide.events,
                   "summary": guide.summary()}, f, indent=1, default=int)
    return rows


def replay_table(run, games, log=print):
    """Avoidable waste and masking risk per game, pooled over seeds and over levels 2 and up:
    <run>/s5_act/replay_summary.{json,md}."""
    per = {}
    for game in games:
        rows = []
        for p in sorted(glob.glob(os.path.join(run.stage_dir("s5_act"), f"{game}_s*.replay.json"))):
            with open(p) as f:
                r = json.load(f)
            rows += [dict(x, seed=r["seed"]) for x in r["levels"] if x["level"] >= 2]
        win = sum(x["window"] for x in rows)
        per[game] = {"seeds": sorted({x["seed"] for x in rows}), "levels": len(rows), "window": win,
                     "covered": sum(x["covered"] for x in rows), "waste_moves": sum(x["waste"] for x in rows),
                     "masking_moves": sum(x["masking"] for x in rows),
                     "waste": round(sum(x["waste"] for x in rows) / win, 4) if win else 0.0,
                     "masking": round(sum(x["masking"] for x in rows) / win, 4) if win else 0.0,
                     "by": dict(sum((collections.Counter(x["by"]) for x in rows), collections.Counter()))}
    lines = [f"# S5 replay: {run.name}", "",
             "Levels 2 and up, the first actions of each level (waste_window), pooled over seeds and levels.", "",
             "| Game | Seeds | Levels | Moves | Covered | Avoidable waste | Masking risk | Covered by |",
             "|---|---|---|---|---|---|---|---|"]
    for g, r in per.items():
        lines.append(f"| {g} | {','.join(map(str, r['seeds']))} | {r['levels']} | {r['window']:,} | {r['covered']:,} | "
                     f"{r['waste']:.1%} ({r['waste_moves']:,}) | {r['masking']:.2%} ({r['masking_moves']:,}) | "
                     + (", ".join(f"{k} {v:,}" for k, v in sorted(r["by"].items())) or "-") + " |")
    text = "\n".join(lines) + "\n"
    d = run.stage_dir("s5_act")
    with open(os.path.join(d, "replay_summary.json"), "w") as f:
        json.dump(per, f, indent=1)
    with open(os.path.join(d, "replay_summary.md"), "w") as f:
        f.write(text)
    log(text)
    return per
