"""E21: E20 extended to games played with keyboard actions and moving objects.

E20 only understood click puzzles with objects at fixed positions (ft09, lp85). On the 8 held-out games it learned
nothing and reduced to StochasticGoose. E21 keeps E20 unchanged for click-only games and adds, for every game that has
non-click actions, generic machinery that again learns everything it knows from its own play:

  Perception      single-colour connected components ("parts"); a kind = colour + scale-normalised shape.
                  Clock lines: frame-edge rows/columns that change on most steps whatever the action (a step counter
                  or energy bar); they are masked out when deciding whether two frames are the same state.
  World graph     every masked frame is a node; every action tried from it is an edge to the frame that followed
                  (or to DEATH if the game ended). Untried actions are the frontier. The agent tries an untried
                  action at the current node, otherwise walks the known graph to the cheapest node that still has
                  one. When the graph is exhausted it falls back to StochasticGoose.
  Movers          parts of a kind that appears exactly once before and after a keyboard action and changed position:
                  which kind each action moves, and by how much.

  Knowledge store (JSON, written at every level clear, used on every later level)
      KA  action effects   per action / clicked kind: how often it changed the state -> order of untried actions
      KM  mover            the kind the agent controls and each action's displacement
      KC  clock lines      which frame edges are a clock
      KG  goal kinds       kinds the controlled object was touching when the level ended (positive example),
                           ranked by how rarely it touched them on all other visited states (negatives);
                           confirmed on each later level that ends the same way
      KD  consumed kinds   kinds whose count dropped over the level
      KT  ending actions   the action (or clicked kind) that ended each level
  Use on a new level: frontier nodes are ranked by path length + distance from the controlled object to the
  best-ranked goal kind (or consumed kind), and untried actions are ordered by learned effect rates and by whether the
  learned displacement moves the controlled object toward that kind. Actions whose learned effect rate is low are
  only tried once nothing better is left anywhere in the graph; level-ending clicked kinds are tried first.

What I designed (generic priors, not game knowledge): the perception above, "a game state is a frame", "objects that
move under an action are controlled", "touching an object can matter", and the graph search. What is learned per game:
every entry of the knowledge store.
"""
import argparse, hashlib, json, os, random
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd
from arc_agi import Arcade
from arcengine import GameAction

import e18_knowledge_planner as e18
from e18_knowledge_planner import StochasticGooseAgent, get_grid, normalised_pattern
from e20_learned_knowledge import LearnedKnowledgeAgent

DEATH = "DEATH"


# =====================================================================================================================
# Perception
# =====================================================================================================================
class Part:
    __slots__ = ("kind", "colour", "pos", "box", "area", "click")

    def __init__(self, ys, xs, colour):
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        patch = np.full((y1 - y0 + 1, x1 - x0 + 1), -1, dtype=int)
        patch[ys - y0, xs - x0] = colour
        pat, _ = normalised_pattern(patch)
        self.kind = f"c{colour}:{pat.shape[0]}x{pat.shape[1]}:{hashlib.md5((pat >= 0).tobytes()).hexdigest()[:8]}"
        self.colour = int(colour)
        self.pos = ((y0 + y1) / 2.0, (x0 + x1) / 2.0)
        self.box = (y0, y1, x0, x1)
        self.area = len(ys)
        i = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
        self.click = (int(xs[i]), int(ys[i]))


def background_colours(grid, big_frac=0.08):
    h, w = grid.shape
    bg = set()
    for c in np.unique(grid):
        for ys, xs in e18._components(grid == c):
            if len(ys) > big_frac * h * w:
                bg.add(int(c)); break
    return bg


def parse_parts(grid, bg):
    parts = []
    for c in np.unique(grid):
        if int(c) in bg:
            continue
        for ys, xs in e18._components(grid == c):
            parts.append(Part(ys, xs, int(c)))
    return parts


def box_gap(a, b):
    """Chebyshev gap between two boxes (0 = touching or overlapping)."""
    dy = max(0, max(a[0], b[0]) - min(a[1], b[1]) - 1)
    dx = max(0, max(a[2], b[2]) - min(a[3], b[3]) - 1)
    return max(dy, dx)


class ClockDetector:
    """Frame-edge lines (2 outermost rows/cols on each side) that change on most steps are a clock."""
    LINES = [("r", 0), ("r", 1), ("r", -1), ("r", -2), ("c", 0), ("c", 1), ("c", -1), ("c", -2)]

    def __init__(self, known=None):
        self.steps, self.counts = 0, Counter()
        self.lines = set(tuple(l) for l in (known or []))

    def observe(self, pre, post):
        if pre.shape != post.shape:
            return False
        d = pre != post
        self.steps += 1
        for kind, i in self.LINES:
            if (d[i, :] if kind == "r" else d[:, i]).any():
                self.counts[(kind, i)] += 1
        if self.steps >= 12:
            new = {l for l in self.LINES if self.counts[l] >= 0.6 * self.steps}
            if new - self.lines:
                self.lines |= new
                return True                          # mask changed
        return False

    def mask(self, grid):
        g = grid.copy()
        for kind, i in self.lines:
            if kind == "r":
                g[i, :] = -1
            else:
                g[:, i] = -1
        return g


# =====================================================================================================================
# Knowledge store
# =====================================================================================================================
class ExplorerKnowledge:
    def __init__(self):
        self.action_stats = {}       # action key -> [tries, changed]
        self.mover = Counter()       # kind -> number of moves seen
        self.disp = defaultdict(Counter)  # action -> Counter("dy,dx") for the controlled kind
        self.clock = []
        self.goal = {}               # kind -> dict(confirmed, neg_freq, levels)
        self.consumed = {}           # kind -> levels seen consumed
        self.trigger = Counter()     # action key (clicks pooled by kind) of the action that ended a level
        self.history = []

    def controlled(self):
        return self.mover.most_common(1)[0][0] if self.mover else None

    def displacement(self, akey):
        c = self.disp.get(akey)
        if not c:
            return None
        (s, n), = c.most_common(1)
        dy, dx = map(int, s.split(","))
        return (dy, dx) if n >= 2 else None

    def goal_ranked(self):
        return sorted(self.goal, key=lambda k: (-self.goal[k]["confirmed"], self.goal[k]["neg_freq"], k))

    def to_json(self):
        return {
            "KA_action_effects": {k: dict(tries=v[0], changed=v[1]) for k, v in sorted(self.action_stats.items())},
            "KM_controlled_kind": self.controlled(),
            "KM_mover_counts": dict(self.mover.most_common(8)),
            "KM_displacement": {a: self.displacement(a) for a in sorted(self.disp)},
            "KC_clock_lines": [list(l) for l in self.clock],
            "KG_goal_kinds_ranked": [dict(rank=i + 1, kind=k, **self.goal[k]) for i, k in enumerate(self.goal_ranked())],
            "KD_consumed_kinds": self.consumed,
            "KT_level_ending_actions": dict(self.trigger),
            "level_history": self.history,
        }


# =====================================================================================================================
# The agent
# =====================================================================================================================
class ObjectExplorerAgent(LearnedKnowledgeAgent):
    name = "E21_object_explorer"

    def __init__(self, seed=0, knowledge_path=None, max_clicks=24, bfs_nodes=20000, goal_weight=1.0,
                 min_prior=0.3, transfer=True, **kw):
        super().__init__(seed=seed, knowledge_path=knowledge_path, **kw)
        self.transfer = transfer                 # False: ablation, the knowledge store is wiped at every level start
        if not transfer:
            self.name = "E21b_no_transfer"
        self.max_clicks, self.bfs_nodes, self.goal_weight, self.min_prior = max_clicks, bfs_nodes, goal_weight, min_prior

    def reset(self):
        super().reset()
        self.click_only = None
        self.xk = ExplorerKnowledge()
        self.rng = random.Random(self.seed)
        self._x_begin_level()

    def _x_begin_level(self):
        if not self.transfer:
            self.xk = ExplorerKnowledge()
        self.clock = ClockDetector(self.xk.clock)
        self.bg = None
        self.graph = defaultdict(dict)          # node -> {akey: Counter(next node)}
        self.node_info = {}                     # node -> dict(cands, goal_dist, contacts)
        self.last = None                        # (node, akey, grid, parts) of the previous action
        self.route, self.route_expect = [], None  # [(akey, node expected after it)]
        self.level_parts0 = None
        self.contact_log = []                   # contact sets of every visited state
        self.x_mode = "explore"
        self.x_log = Counter()

    # -- lifecycle -----------------------------------------------------------------------------------------------------
    def note(self, msg):
        super().note(msg)
        if self.click_only is False and "GAME_OVER" in msg and self.last is not None:
            node, akey = self.last[0], self.last[1]
            self.graph[node].setdefault(akey, Counter())[DEATH] += 1
            self.last, self.route = None, []
            self.x_log["deaths"] += 1

    def act(self, obs, legal):
        if self.click_only is None:
            self.click_only = all(a.is_complex() for a in legal)
        if self.click_only:
            return super().act(obs, legal)             # E20, unchanged
        return self._x_act(obs, legal)

    # -- keys ----------------------------------------------------------------------------------------------------------
    def _node(self, grid):
        return hashlib.md5(self.clock.mask(grid).tobytes()).hexdigest()[:16]

    def _candidates(self, parts, legal):
        cands = [a.name for a in legal if not a.is_complex()]
        if any(a.is_complex() for a in legal):
            seen, clicks = set(), []
            for p in sorted(parts, key=lambda p: p.area):
                key = (p.kind, p.click)
                if key in seen:
                    continue
                seen.add(key); clicks.append(f"click:{p.kind}@{p.click[0]},{p.click[1]}")
            cands += clicks[: self.max_clicks]
        return cands

    @staticmethod
    def _stat_key(akey):
        return akey.split("@")[0]               # clicks are pooled by kind

    def _avatar(self, parts):
        k = self.xk.controlled()
        if k is None:
            return None
        ps = [p for p in parts if p.kind == k]
        return ps[0] if len(ps) == 1 else None

    def _targets(self, parts):
        for k in self.xk.goal_ranked():
            ps = [p for p in parts if p.kind == k]
            if ps:
                return ps
        cons = sorted(self.xk.consumed, key=lambda k: -self.xk.consumed[k])
        for k in cons:
            ps = [p for p in parts if p.kind == k]
            if ps:
                return ps
        return []

    def _goal_dist(self, parts):
        av = self._avatar(parts)
        tg = self._targets(parts)
        if av is None or not tg:
            return 0.0
        step = max([max(abs(d[0]), abs(d[1])) for d in (self.xk.displacement(a) for a in list(self.xk.disp)) if d]
                   or [1])
        return min(box_gap(av.box, t.box) for t in tg) / step

    def _contacts(self, parts, tol=None):
        av = self._avatar(parts)
        if av is None:
            return set()
        if tol is None:
            ds = [d for d in (self.xk.displacement(a) for a in list(self.xk.disp)) if d]
            tol = max([max(abs(d[0]), abs(d[1])) for d in ds] or [1])
        return {p.kind for p in parts if p is not av and box_gap(av.box, p.box) <= tol}

    # -- learning from one transition ----------------------------------------------------------------------------------
    def _learn_transition(self, akey, pre_grid, pre_parts, post_grid, post_parts, level_end=False):
        changed = bool((self.clock.mask(pre_grid) != self.clock.mask(post_grid)).any()) \
            if pre_grid.shape == post_grid.shape else True
        st = self.xk.action_stats.setdefault(self._stat_key(akey), [0, 0])
        st[0] += 1; st[1] += int(changed)
        if not changed or akey.startswith("click:"):
            return
        before, after = defaultdict(list), defaultdict(list)
        for p in pre_parts:
            before[p.kind].append(p)
        for p in post_parts:
            after[p.kind].append(p)
        for k, ps in before.items():
            qs = after.get(k, [])
            if len(ps) == 1 and len(qs) == 1 and ps[0].area == qs[0].area:
                dy, dx = round(qs[0].pos[0] - ps[0].pos[0]), round(qs[0].pos[1] - ps[0].pos[1])
                if (dy, dx) != (0, 0):
                    self.xk.mover[k] += 1
                    if k == self.xk.controlled():
                        self.xk.disp[akey][f"{dy},{dx}"] += 1

    # -- level clear: induce goal kinds and write the store --------------------------------------------------------------
    def _x_extract(self, win_grid):
        pre_parts = self.last[3] if self.last is not None else None
        if self.last is not None and self.last[1]:
            self.xk.trigger[self._stat_key(self.last[1])] += 1
        positive = set()
        if pre_parts is not None:
            positive |= self._contacts(pre_parts)
        if win_grid is not None and self.bg is not None:
            positive |= self._contacts(parse_parts(win_grid, self.bg))
        negatives = self.contact_log[:-1] or [set()]
        for k in positive:
            freq = sum(k in c for c in negatives) / len(negatives)
            s = self.xk.goal.setdefault(k, dict(confirmed=0, neg_freq=freq, levels=[]))
            s["confirmed"] += 1; s["levels"].append(self.level + 1)
            s["neg_freq"] = round((s["neg_freq"] * (s["confirmed"] - 1) + freq) / s["confirmed"], 3)
        if self.level_parts0 is not None and pre_parts is not None:
            c0 = Counter(p.kind for p in self.level_parts0)
            c1 = Counter(p.kind for p in pre_parts)
            for k, n in c0.items():
                if c1.get(k, 0) < n and k != self.xk.controlled():
                    self.xk.consumed[k] = self.xk.consumed.get(k, 0) + 1
        self.xk.clock = sorted(self.clock.lines)
        self.xk.history.append(dict(level=self.level + 1, actions=self.level_actions, states=len(self.graph),
                                    deaths=self.x_log["deaths"], goose_steps=self.x_log["goose"],
                                    contacts_at_win=sorted(positive)))
        if self.knowledge_path:
            os.makedirs(os.path.dirname(self.knowledge_path), exist_ok=True)
            with open(self.knowledge_path, "w") as f:
                json.dump(self.xk.to_json(), f, indent=1)

    # -- acting --------------------------------------------------------------------------------------------------------
    def _x_act(self, obs, legal):
        grid = get_grid(obs)
        if obs.levels_completed > self.level:
            f = np.array(obs.frame)
            win_grid = f[-2].astype(int) if f.ndim == 3 and len(f) >= 2 else None
            if self.last is not None:                  # counts the level-ending action's effect
                self._learn_transition(self.last[1], self.last[2], self.last[3], grid, parse_parts(grid, self.bg),
                                       level_end=True)
            self._x_extract(win_grid)
            self._check_level_up(obs)
            self._x_begin_level()
        if self.bg is None:
            self.bg = background_colours(grid)
        parts = parse_parts(grid, self.bg)
        if self.level_parts0 is None:
            self.level_parts0 = parts

        if self.last is not None:
            pnode, akey, pgrid, pparts = self.last
            if self.clock.observe(pgrid, grid):        # clock found: re-key the graph from scratch
                self.graph.clear(); self.node_info.clear(); self.route = []
            else:
                self.graph[pnode].setdefault(akey, Counter())[self._node(grid)] += 1
            self._learn_transition(akey, pgrid, pparts, grid, parts)

        node = self._node(grid)
        if node not in self.node_info:
            self.node_info[node] = dict(cands=self._candidates(parts, legal), goal_dist=self._goal_dist(parts))
        self.contact_log.append(self._contacts(parts))

        akey = self._pick(node, parts, grid, legal)
        if akey is None:                                # graph exhausted: StochasticGoose
            self.x_log["goose"] += 1
            action, data, raw, ok = StochasticGooseAgent.act(self, obs, legal)
            akey = action.name if not action.is_complex() else None
            self.last = (node, akey, grid, parts) if akey else None
            return action, data, raw, ok
        self.last = (node, akey, grid, parts)
        self.level_actions += 1; self.step_count += 1
        if akey.startswith("click:"):
            x, y = map(int, akey.split("@")[1].split(","))
            act = next(a for a in legal if a.is_complex())
            return act, {"x": x, "y": y}, f"{self.name}:{akey}", True
        return GameAction[akey], {}, f"{self.name}:{akey}", True

    def _pick(self, node, parts, grid, legal):
        """Which action to take now (subclasses may plan instead of exploring)."""
        return self._choose(node, parts)

    def _untried(self, node, tier=1):
        """Tier 1: untried actions whose learned effect rate is worth a try. Tier 2: all untried actions."""
        u = [a for a in self.node_info[node]["cands"] if a not in self.graph[node]]
        return [a for a in u if self._prior(a) >= self.min_prior] if tier == 1 else u

    def _prior(self, akey):
        k = self._stat_key(akey)
        t, c = self.xk.action_stats.get(k, (0, 0))
        ended = k in self.xk.trigger and self.xk.displacement(akey) is None   # a move ending a level says little
        return (c + 1) / (t + 2) + (0.5 if ended else 0.0)

    def _order(self, untried, parts):
        """Highest learned effect rate first; among keyboard actions prefer moving the controlled object toward the
        current target; random tie-break."""
        av, tg = self._avatar(parts), self._targets(parts)

        def toward(a):
            d = self.xk.displacement(a)
            if av is None or not tg or d is None:
                return 0.0
            moved = (av.box[0] + d[0], av.box[1] + d[0], av.box[2] + d[1], av.box[3] + d[1])
            return min(box_gap(av.box, t.box) for t in tg) - min(box_gap(moved, t.box) for t in tg)

        return max(untried, key=lambda a: (round(self._prior(a), 1), toward(a), self.rng.random()))

    def _choose(self, node, parts):
        if self.route and self.route_expect == node:          # walking a known path to a frontier node
            akey, self.route_expect = self.route.pop(0)
            self.x_log["walk"] += 1
            return akey
        self.route = []
        path = None
        for tier in (1, 2):
            u = self._untried(node, tier)
            if u:
                self.x_log["explore"] += 1
                return self._order(u, parts)
            path = self._route_to_frontier(node, tier)
            if path:
                break
        if not path:
            return None
        self.route = path
        akey, self.route_expect = self.route.pop(0)
        self.x_log["walk"] += 1
        return akey

    def _route_to_frontier(self, start, tier=1):
        """BFS over the known graph (most frequent successor of each edge, never edges that mostly died) to the node
        minimising path length + goal_weight * distance to the target kind."""
        prev = {start: None}
        q = deque([(start, 0)])
        best, best_cost = None, float("inf")
        while q and len(prev) < self.bfs_nodes:
            n, d = q.popleft()
            if d >= best_cost:
                continue
            if n != start and n in self.node_info and self._untried(n, tier):
                cost = d + self.goal_weight * self.node_info[n]["goal_dist"]
                if cost < best_cost:
                    best, best_cost = n, cost
            for a, succ in self.graph.get(n, {}).items():
                (m, _), = succ.most_common(1)
                if m == DEATH or m in prev:
                    continue
                prev[m] = (n, a)
                q.append((m, d + 1))
        if best is None:
            return None
        path, n = [], best
        while prev[n] is not None:
            p, a = prev[n]
            path.append((a, n)); n = p
        return path[::-1]

    def stats(self):
        s = super().stats()
        if self.click_only is False:
            s["knowledge_ranked"] = str(self.xk.goal_ranked()[:3])
            s["plan_log"] = str(dict(self.x_log))
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--games", required=True)
    ap.add_argument("--knowledge-dir", default="results/knowledge_e21")
    args = ap.parse_args()
    arc = Arcade()
    for seed in [int(s) for s in args.seeds.split(",")]:
        for g in args.games.split(","):
            agent = ObjectExplorerAgent(seed=seed, knowledge_path=os.path.join(args.knowledge_dir, f"{g}_s{seed}.json"))
            sc = arc.create_scorecard(tags=["E21", f"s{seed}"])
            r = e18.run_episode(arc, g, agent, sc, seed=seed)
            print(f"E21 s{seed} {g}: levels={r['levels']} clears={r['level_clears_log']} t={r['seconds']}s "
                  f"{r.get('plan_log')}", flush=True)


if __name__ == "__main__":
    main()
