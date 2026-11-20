"""E20: the agent learns what it knows about a game from its own play, stores it, and carries it level to level.

No game-specific goal descriptions. The agent induces its goal hypothesis from data:

  Hypothesis space (generic, compositional; the same for every game)
      quantifier   AA  every related pair satisfies the colour relation      (gives per-object constraints)
                   AE  every subject has at least one related object satisfying it
                   DE  every dynamic object has at least one related subject satisfying it
                   G   global property of the dynamic objects
      subject      mark     static single-colour objects of one shape, whose colour also occurs on dynamic objects
                   frame    rectangle closed by four identical marks (Gestalt closure)
                   pattern  static multi-colour odd-square object, read cell-by-cell as a map of its surroundings
      relation     nearest, adj8, row, col (marks) | inside (frames) | cell (patterns)
      colour rel.  same / diff as the subject's colour | per-symbol relation learned from data (patterns)
      global       uniform colour; no two 4-adjacent dynamic objects share a colour

  Induction (at every level clear)
      positive example  the solved frame
      negative examples every other state seen on that level (the level would have ended if the goal held there)
      A hypothesis is consistent on a level if it holds in the solved frame and in none of the negatives.

  Knowledge store (saved as JSON at every level clear, read at the start of every later level)
      K1  which object kinds did something when clicked
      K2  dynamics family (appearance cycle vs. movement between slots) and cycle length -- a guess to re-verify
      K3  every goal hypothesis with confirmed / refuted counts, and learned symbol relations
      level history: actions used, which hypothesis cleared the level

On a new level the agent probes K1 kinds and never-seen kinds, verifies its effect model (E19), grounds the
best-ranked surviving hypothesis, plans, and refutes it (moving to the next) if the plan reaches a state the hypothesis
calls solved but the level does not end. Level 1 is played by StochasticGoose (E0), since nothing is known yet.
"""
import argparse, json, os
from collections import Counter, defaultdict
from itertools import product

import pandas as pd
from arc_agi import Arcade

import e18_knowledge_planner as e18
from e18_knowledge_planner import Grounded, SigTable, _pitch, cycle_period, infer_dynamics_class
from e19_adaptive_planner import AdaptivePlannerAgent, legend_ok


# =====================================================================================================================
# Perception helpers
# =====================================================================================================================
class MajoritySigTable(SigTable):
    """A multi-colour object's colour is its majority pixel colour (if there is a strict majority)."""

    def intern(self, o):
        new = o.sig not in self.ids
        i = super().intern(o)
        if new and o.colour is None:
            counts = Counter(o.pattern[o.pattern >= 0].tolist()).most_common()
            if counts and (len(counts) == 1 or counts[0][1] > counts[1][1]):
                self.colour_of[i] = int(counts[0][0])
        return i


def shape_key(o):
    (h, w), mask = o.shape
    return f"{h}x{w}:{mask.hex()}"


class LevelContext:
    """Everything a hypothesis needs to ground itself on one level's layout."""

    def __init__(self, objs, slots, dyn, static, colour_of, dyn_colours, parts=()):
        self.objs, self.slots, self.colour_of = objs, slots, colour_of
        self.dyn, self.static = list(dyn), list(static)
        self.parts = list(parts)          # single-colour pieces of static multi-colour objects (touching sprites)
        self.dyn_colours = set(dyn_colours) - {None}
        self.pitch = _pitch([slots[i] for i in self.dyn]) if len(self.dyn) >= 2 else 1.0

    def mark_objs(self, shape=None):
        cands = [self.objs[i] for i in self.static] + self.parts
        return [o for o in cands if o.colour is not None and o.colour in self.dyn_colours
                and (shape is None or shape_key(o) == shape)]

    def mark_shapes(self):
        return {shape_key(o) for o in self.mark_objs()}

    def frames(self, shape):
        """Rectangles closed by four identical marks: smallest first, each mark used once, no mark inside."""
        by = defaultdict(set)
        for o in self.mark_objs(shape):
            by[o.sig].add((o.pos, o.colour))
        rects = []
        for marks in by.values():
            pts = {p for p, _ in marks}
            col = next(iter(marks))[1]
            for (y0, x0) in pts:
                for (y1, x1) in pts:
                    if y1 > y0 and x1 > x0 and (y0, x1) in pts and (y1, x0) in pts:
                        corners = {(y0, x0), (y0, x1), (y1, x0), (y1, x1)}
                        if any(y0 <= py <= y1 and x0 <= px <= x1 for py, px in pts - corners):
                            continue
                        rects.append(((y0, x0, y1, x1), col, corners))
        used, out = set(), []
        for bbox, col, corners in sorted(rects, key=lambda r: (r[0][2] - r[0][0]) * (r[0][3] - r[0][1])):
            if corners & used:
                continue
            used |= corners
            out.append((bbox, col))
        return out

    def patterns(self):
        out = []
        for i in self.static:
            o = self.objs[i]
            k = o.pattern.shape[0]
            if o.colour is None and o.pattern.shape == (k, k) and k >= 3 and k % 2 == 1 and not (o.pattern < 0).any():
                out.append(o)
        return out

    def dyn_near(self, y, x, tol=None):
        tol = self.pitch / 3 if tol is None else tol
        best = [(abs(self.slots[p][0] - y) + abs(self.slots[p][1] - x), p) for p in self.dyn]
        best = [b for b in best if b[0] <= tol]
        return min(best)[1] if best else None

    def related(self, pos, rel):
        """Dynamic slots related to a subject at `pos`."""
        sy, sx = pos
        cheb = {p: max(abs(self.slots[p][0] - sy), abs(self.slots[p][1] - sx)) for p in self.dyn}
        if rel == "nearest":
            m = min(cheb.values(), default=None)
            return [p for p, d in cheb.items() if m is not None and d <= m + 0.5]
        if rel == "adj8":
            return [p for p, d in cheb.items() if 0 < d <= 1.2 * self.pitch]
        if rel == "row":
            return [p for p in self.dyn if abs(self.slots[p][0] - sy) < self.pitch / 3]
        if rel == "col":
            return [p for p in self.dyn if abs(self.slots[p][1] - sx) < self.pitch / 3]
        raise ValueError(rel)


def static_parts(layout, dynamic):
    """Single-colour pieces of non-moving multi-colour objects: touching sprites seen as separate objects."""
    cells = set()
    for i in (i for i in range(len(layout.objs)) if i not in set(dynamic)):
        o = layout.objs[i]
        if o.colour is None:
            cells |= set(zip(o.ys.tolist(), o.xs.tolist()))
    return [pt for pt in layout.parts if (int(pt.ys[0]), int(pt.xs[0])) in cells]


def colour_rel_ok(colour, crel, ref):
    return colour == ref if crel == "same" else colour != ref


# =====================================================================================================================
# Lifted hypotheses
# =====================================================================================================================
class Hyp:
    def __init__(self, quant, subj, shape, rel, crel):
        self.quant, self.subj, self.shape, self.rel, self.crel = quant, subj, shape, rel, crel
        self.key = f"{quant}|{subj}|{shape}|{rel}|{crel}"
        self.complexity = ({"AA": 1, "AE": 2, "DE": 2, "G": 1}[quant] + (subj in ("frame", "pattern"))
                           + (0.5 if crel == "diff" else 0) + {"nearest": 0, "adj8": 0.2, "row": 0.4, "col": 0.4}.get(rel, 0))

    def describe(self):
        if self.subj == "pattern":
            return "every multi-colour pattern describes its neighbourhood: each symbol fixes a colour relation"
        if self.quant == "G":
            return {"uniform": "all dynamic objects share one colour",
                    "adj4": "no two adjacent dynamic objects share a colour"}[self.rel]
        q = {"AA": "for every {s}, every {r} dynamic object has the {c} colour",
             "AE": "for every {s}, some {r} dynamic object has the {c} colour",
             "DE": "every dynamic object has a {s} it is {r} with the {c} colour"}[self.quant]
        return q.format(s=self.subj, r={"inside": "enclosed", "nearest": "nearest", "adj8": "adjacent",
                                        "row": "same-row", "col": "same-column"}[self.rel],
                        c={"same": "same", "diff": "a different"}[self.crel])

    # -- grounding ---------------------------------------------------------------------------------------------------
    def subjects(self, ctx):
        """[(related dynamic slots, reference colour)] for every subject on this level."""
        if self.subj == "mark":
            return [(ctx.related(o.pos, self.rel), o.colour) for o in ctx.mark_objs(self.shape)]
        if self.subj == "frame":
            return [([p for p in ctx.dyn if y0 < ctx.slots[p][0] < y1 and x0 < ctx.slots[p][1] < x1], col)
                    for (y0, x0, y1, x1), col in ctx.frames(self.shape)]
        return []

    def pattern_cells(self, ctx):
        """[(slot, symbol, centre colour)] for every pattern cell that lands on a dynamic object."""
        cells = []
        for o in ctx.patterns():
            k = o.pattern.shape[0]
            c = k // 2
            centre = int(o.pattern[c, c])
            for r in range(k):
                for cc in range(k):
                    if (r, cc) == (c, c):
                        continue
                    p = ctx.dyn_near(o.pos[0] + (r - c) * ctx.pitch, o.pos[1] + (cc - c) * ctx.pitch)
                    if p is not None:
                        cells.append((p, int(o.pattern[r, cc]), centre))
        return cells

    def learn_mapping(self, ctx, win_state):
        """Per symbol: the relation (same / diff as centre, or one absolute colour) that holds at every occurrence."""
        obs = defaultdict(list)
        for p, v, centre in self.pattern_cells(ctx):
            obs[v].append((centre, ctx.colour_of[win_state[p]]))
        mapping = {}
        for v, pairs in obs.items():
            if all(t == c for c, t in pairs):
                mapping[v] = ("same",)
            elif all(t != c for c, t in pairs):
                mapping[v] = ("diff",)
            elif len({t for _, t in pairs}) == 1:
                mapping[v] = ("abs", pairs[0][1])
            else:
                return None
        return mapping or None

    def ground(self, ctx, mapping=None):
        colour_of = ctx.colour_of
        if self.quant == "G":
            dyn = ctx.dyn
            if len(dyn) < 2:
                return None
            if self.rel == "uniform":
                test = lambda s: len({colour_of[s[p]] for p in dyn}) == 1
                heur = lambda s: float(len(dyn) - max(Counter(colour_of[s[p]] for p in dyn).values()))
            else:
                pairs = [(p, q) for p in dyn for q in dyn if p < q and
                         abs(ctx.slots[p][0] - ctx.slots[q][0]) + abs(ctx.slots[p][1] - ctx.slots[q][1]) <= 1.2 * ctx.pitch]
                if not pairs:
                    return None
                test = lambda s: all(colour_of[s[p]] != colour_of[s[q]] for p, q in pairs)
                heur = lambda s: float(sum(colour_of[s[p]] == colour_of[s[q]] for p, q in pairs))
            return Grounded(self.key, test, heur, self.describe())

        if self.subj == "pattern":
            cells = self.pattern_cells(ctx)
            if not cells or not mapping or any(v not in mapping for _, v, _ in cells):
                return None
            conds = [(p, mapping[v], centre) for p, v, centre in cells]
            g = Grounded(self.key, lambda s: all(legend_ok(colour_of[s[p]], r, c) for p, r, c in conds),
                         lambda s: float(sum(not legend_ok(colour_of[s[p]], r, c) for p, r, c in conds)),
                         f"{len(conds)} pattern cells, symbols {mapping}")
            g.conds = conds
            return g

        subs = self.subjects(ctx)
        if not subs or any(not rel for rel, _ in subs):
            return None
        crel = self.crel
        if self.quant == "AA":
            conds = [(p, (crel,), col) for rel, col in subs for p in rel]
            # with "same", only objects of the subjects' colours matter -> exact projected search under movement
            g = Grounded(self.key, lambda s: all(legend_ok(colour_of[s[p]], r, c) for p, r, c in conds),
                         lambda s: float(sum(not legend_ok(colour_of[s[p]], r, c) for p, r, c in conds)),
                         f"{len(subs)} {self.subj}(s)",
                         relevant_colours={c for _, c in subs} if crel == "same" else None)
            g.conds = conds
            return g
        if self.quant == "AE":
            test = lambda s: all(any(colour_rel_ok(colour_of[s[p]], crel, col) for p in rel) for rel, col in subs)
            heur = lambda s: float(sum(not any(colour_rel_ok(colour_of[s[p]], crel, col) for p in rel)
                                       for rel, col in subs))
            return Grounded(self.key, test, heur, f"{len(subs)} {self.subj}(s), colours {sorted({c for _, c in subs})}",
                            relevant_colours={c for _, c in subs} if crel == "same" else None)
        # DE: every dynamic object related to some subject has a satisfying subject
        rel_of = defaultdict(list)
        for rel, col in subs:
            for p in rel:
                rel_of[p].append(col)
        if set(rel_of) != set(ctx.dyn):
            return None
        test = lambda s: all(any(colour_rel_ok(colour_of[s[p]], crel, c) for c in cols) for p, cols in rel_of.items())
        heur = lambda s: float(sum(not any(colour_rel_ok(colour_of[s[p]], crel, c) for c in cols)
                                   for p, cols in rel_of.items()))
        return Grounded(self.key, test, heur, f"{len(rel_of)} dynamic objects")


def enumerate_hyps(ctx):
    hyps = []
    for shape in sorted(ctx.mark_shapes()):
        for rel, crel, q in product(["nearest", "adj8", "row", "col"], ["same", "diff"], ["AA", "AE", "DE"]):
            hyps.append(Hyp(q, "mark", shape, rel, crel))
        if ctx.frames(shape):
            for crel, q in product(["same", "diff"], ["AA", "AE"]):
                hyps.append(Hyp(q, "frame", shape, "inside", crel))
    if ctx.patterns():
        hyps.append(Hyp("AA", "pattern", None, "cell", "learned"))
    hyps += [Hyp("G", "dyn", None, "uniform", None), Hyp("G", "dyn", None, "adj4", "diff")]
    return hyps


# =====================================================================================================================
# The knowledge store
# =====================================================================================================================
class KnowledgeStore:
    def __init__(self):
        self.hyps = {}          # key -> Hyp
        self.stats = {}         # key -> dict(confirmed, refuted, levels, mapping, n_constraints)
        self.k1 = {}            # shape key -> dict(clicks, changed)
        self.k2 = {}            # family, period, per level
        self.history = []       # per level: actions, hypothesis that cleared it

    def ranked(self):
        ok = [k for k, s in self.stats.items() if s["confirmed"] > 0 and s["refuted"] == 0]
        return sorted(ok, key=lambda k: (-self.stats[k]["confirmed"], self.hyps[k].complexity,
                                         -self.stats[k]["n_constraints"], k))

    def to_json(self):
        ranked = self.ranked()
        return {
            "K1_interactive_kinds": self.k1,
            "K2_dynamics": self.k2,
            "K3_goal_hypotheses_ranked": [dict(rank=i + 1, key=k, meaning=self.hyps[k].describe(), **{
                kk: (str(v) if kk == "mapping" else v) for kk, v in self.stats[k].items()}) for i, k in enumerate(ranked)],
            "K3_rejected_count": sum(1 for s in self.stats.values() if s["refuted"] > 0 or s["confirmed"] == 0),
            "level_history": self.history,
        }


# =====================================================================================================================
# The agent
# =====================================================================================================================
class Candidate:
    """One concrete goal to try on the current level: a hypothesis plus, for patterns, a symbol mapping."""

    def __init__(self, hyp, mapping=None):
        self.hyp, self.mapping, self.name = hyp, mapping, hyp.key


class LearnedKnowledgeAgent(AdaptivePlannerAgent):
    name = "E20_learned_knowledge"

    def __init__(self, seed=0, knowledge_path=None, max_negatives=800, **kw):
        super().__init__(seed=seed, **kw)
        self.knowledge_path = knowledge_path
        self.max_negatives = max_negatives

    def reset(self):
        super().reset()
        self.sigs = MajoritySigTable()
        self.store = KnowledgeStore()

    def _begin_level(self, grid):
        super()._begin_level(grid)
        self.candidates = None

    # -- learning at a level clear -------------------------------------------------------------------------------------
    def _extract_knowledge(self, win_grid):
        lay = self.layout
        if lay is None:
            return
        st = self.store
        # K1: object kinds whose click changed something
        for akey, pre, post in self.level_transitions:
            i = lay.slot_index.get(akey)
            if i is None:
                continue
            o = lay.objs[i]
            rec = st.k1.setdefault(shape_key(o), {"clicks": 0, "changed": 0})
            rec["clicks"] += 1
            self.k1_clicks[o.shape] += 1
            if pre != post:
                rec["changed"] += 1
                self.k1_changed[o.shape] += 1
                self.k1_sigs.add(o.sig)
        self.k1_shapes |= {s for s in self.k1_clicks if self.k1_changed[s] > 0}
        # K2: dynamics family and cycle length (a guess the next level re-verifies)
        changes = [(pre, post) for _, pre, post in self.level_transitions if pre != post]
        if changes:
            self.k2_class, succ = infer_dynamics_class(changes)
            self.k2_period = cycle_period(succ) if self.k2_class == "recolour" else None
            st.k2[f"level {self.level + 1}"] = {"family": self.k2_class, "cycle_length": self.k2_period}
        # K3: induce goal hypotheses from the solved frame (positive) and every other state seen (negatives)
        win, _ = lay.encode(win_grid) if win_grid is not None else (None, None)
        start, _ = lay.encode(self.level_start_grid)
        if win is not None:
            negs = sorted((s for s in self.visited | {start} if s is not None and s != win), key=hash)
            negs = negs[:: max(1, len(negs) // self.max_negatives)]
            effectful = {a for a, pre, post in self.level_transitions if pre != post}
            dyn = sorted(lay.dynamic_set)
            static = [i for i in range(len(lay.slots)) if i not in lay.dynamic_set and lay.slots[i] not in effectful]
            dyn_colours = {self.sigs.colour_of[s[p]] for s in [win, *negs[:50]] for p in dyn}
            ctx = LevelContext(lay.objs, lay.slots, dyn, static, self.sigs.colour_of, dyn_colours,
                               static_parts(lay, dyn))
            n_ok = 0
            # this level's hypothesis space plus everything already in the store (old hypotheses get re-tested)
            for h in enumerate_hyps(ctx) + list(st.hyps.values()):
                st.hyps.setdefault(h.key, h)
                h = st.hyps[h.key]
                rec = st.stats.setdefault(h.key, {"confirmed": 0, "refuted": 0, "levels": [], "mapping": None,
                                                  "n_constraints": 0})
                if self.level + 1 in rec["levels"]:
                    continue
                mapping = None
                if h.subj == "pattern":
                    learned = h.learn_mapping(ctx, win)
                    if learned is None:
                        rec["refuted"] += 1; rec["levels"].append(self.level + 1); continue
                    old = rec["mapping"] or {}
                    if any(v in old and old[v] != r for v, r in learned.items()):
                        rec["refuted"] += 1; rec["levels"].append(self.level + 1); continue
                    mapping = {**old, **learned}
                g = h.ground(ctx, mapping)
                if g is None:
                    continue                          # not applicable on this level: no evidence either way
                rec["levels"].append(self.level + 1)
                if g.test(win) and not any(g.test(n) for n in negs):
                    rec["confirmed"] += 1
                    rec["n_constraints"] = len(getattr(g, "conds", [])) or len(ctx.dyn)
                    if mapping:
                        rec["mapping"] = mapping
                    n_ok += 1
                else:
                    rec["refuted"] += 1
            self.knowledge_log.append((self.level, "K3", f"{n_ok} consistent hypotheses", f"{len(negs)} negatives"))
        if self.level == 0:
            solved_by = "exploration (StochasticGoose)"
        elif self.mode == "goose":
            solved_by = "exploration after the planner gave up (StochasticGoose)"
        else:
            solved_by = f"planning to goal rule {self.goal.name}" if self.goal is not None else "planning"
        st.history.append({"level": self.level + 1, "actions": self.level_actions, "solved_by": solved_by})
        self.k3 = st.ranked()
        self.knowledge_log.append((self.level, "ranked", self.k3[:3], ""))
        if self.knowledge_path:
            os.makedirs(os.path.dirname(self.knowledge_path), exist_ok=True)
            with open(self.knowledge_path, "w") as f:
                json.dump(st.to_json(), f, indent=1, default=str)

    # -- using the knowledge on a new level ------------------------------------------------------------------------------
    def _goal_ctx(self):
        d = self._goal_layout()
        dyn_colours = {self.sigs.colour_of[s[p]] for s in list(self.visited)[:200] for p in d["dynamic"]}
        return LevelContext(d["objs"], d["slots"], d["dynamic"], d["static"], self.sigs.colour_of, dyn_colours,
                            static_parts(self.layout, d["dynamic"]))

    def _ground_goal(self):
        ctx = self._goal_ctx()
        if self.candidates is None:
            self.candidates = []
            for key in self.store.ranked():
                h, rec = self.store.hyps[key], self.store.stats[key]
                if h.subj != "pattern":
                    self.candidates.append(Candidate(h))
                    continue
                known = rec["mapping"] or {}
                unknown = sorted({v for _, v, _ in h.pattern_cells(ctx)} - set(known))
                for combo in product([("any",), ("diff",), ("same",)], repeat=len(unknown)):
                    self.candidates.append(Candidate(h, {**known, **dict(zip(unknown, combo))}))
                    if len(self.candidates) > 60:
                        break
        for cand in self.candidates:
            if cand in self.goal_tried:
                continue
            g = cand.hyp.ground(ctx, cand.mapping)
            if g is not None:
                return cand, g
            self.goal_tried.append(cand)
        return None, None

    def _plan_act(self, grid, legal):
        out = super()._plan_act(grid, legal)
        if out is None and self.goal is None and self.plan_failures > self.max_plan_failures \
                and self.mispredictions <= self.max_mispredictions \
                and any(c not in self.goal_tried for c in (self.candidates or [])):
            self.plan_failures = 0                   # this hypothesis could not be planned: try the next one
            self.log.append(("next-hypothesis", self.level))
            out = super()._plan_act(grid, legal)
        return out

    def stats(self):
        s = super().stats()
        s["knowledge_ranked"] = str(self.store.ranked()[:5])
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--games", default=",".join(e18.LIVE_GAMES))
    ap.add_argument("--budget", type=int, default=e18.BUDGET)
    ap.add_argument("--out", default=None)
    ap.add_argument("--knowledge-dir", default="results/knowledge")
    args = ap.parse_args()
    arc = Arcade()
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for g in args.games.split(","):
            agent = LearnedKnowledgeAgent(seed=seed, knowledge_path=os.path.join(args.knowledge_dir, f"{g}_s{seed}.json"))
            sc = arc.create_scorecard(tags=["E20", f"s{seed}"])
            r = e18.run_episode(arc, g, agent, sc, max_steps=args.budget, seed=seed)
            r["seed"] = seed
            try:
                arc.close_scorecard(sc)
            except Exception:
                pass
            print(f"{r['agent']:22s} s{seed} {g}: levels={r['levels']} clears={r['level_clears_log']} "
                  f"t={r['seconds']}s\n   top hypotheses: {r['knowledge_ranked']}", flush=True)
            rows.append(r)
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
