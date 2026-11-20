"""E19: E18 with transferred knowledge treated as a hypothesis that each new level must confirm.

E18 stalls on level 4 of both games, for two different reasons (diagnosed in the Working_Results discussion):

  ft09  levels 1-3 cycle tiles through 2 colours, so E18 carries "period 2" forward and uses it in the effect model
        (a->b implies b->a) and in its GF(2) solver. Level 4 cycles through 3 colours (9->8->12->9); the transferred
        rule is wrong there and the GF(2) system cannot express the goal.
  lp85  every button is still a fixed permutation on level 4, but with 17 buttons and many identical tiles one probe
        press per button leaves the permutation ambiguous (0/40 next-state predictions correct after 1 press,
        40/40 after 10). E18 relies on misprediction repair, hits its 40-misprediction cap, and gives up.

Two changes, both of the form "verify on this level, don't assume":

  1. Per-level colour successors (recolour class). The successor table starts empty on every level. The transferred
     cycle period is only a *guess* (period 2 -> after seeing a->b, guess b->a); observed transitions always override
     guesses. Goals are solved with a per-slot solver when every click recolours exactly one tile (true in ft09),
     which handles cycles of any length; a successor that is neither observed nor guessed triggers an exploratory
     click on that tile. Coupled (lights-out-style) systems still use E18's GF(2) solver when the period is 2.
  2. Active verification (move class). After probing, every button is pressed again, round-robin, with its effect
     predicted beforehand, until it has predicted correctly `verify_streak` times in a row (or `verify_cap` presses).
     Mispredictions during verification are expected and do not count towards the give-up cap.

  3. Hypothesis testing for the goal. A key-pixel colour never seen in any solved frame gets no relation from the
     data, so its relation is a hypothesis tried in order ignore -> diff -> same. If a plan reaches a state the
     grounded goal says is solved but the level does not end, that grounding is refuted and the next is tried. A
     relation that wins is kept for later levels.
  4. Interactive objects are not legends. Objects whose click changed something are never read as goal keys, and
     objects of a kind never seen before are probed once (after the K1 matches), so new mechanics such as ft09's
     neighbour-flipping tiles enter the model. Multi-colour tiles are coloured by their centre pixel.
  5. A misprediction during planning sends that button back to verification instead of only counting as a failure.

Everything else -- level 1 (= E0), K1/K2/K3 extraction, goal templates, enclosure planning -- is E18 unchanged.
"""
import argparse
from collections import defaultdict
from itertools import product

import pandas as pd
from arc_agi import Arcade

import e18_knowledge_planner as e18
from e18_knowledge_planner import EffectModel, KnowledgePlannerAgent, LegendTemplate, SigTable, cycle_period


def legend_ok(colour, rel, centre):
    # E18's relation test plus ("any",): a key pixel whose relation is hypothesised to be unconstrained
    if rel[0] == "any":
        return True
    return colour == centre if rel[0] == "same" else (colour != centre if rel[0] == "diff" else colour == rel[1])


e18.legend_ok = legend_ok          # LegendTemplate.ground / _solve_gf2 look the name up in the e18 module


class CentreSigTable(SigTable):
    """Multi-colour odd-square tiles take their centre pixel as their colour (single-colour objects unchanged)."""

    def intern(self, o):
        new = o.sig not in self.ids
        i = super().intern(o)
        if new and o.colour is None:
            pat = o.pattern
            h, w = pat.shape
            if h == w and h % 2 == 1 and pat[h // 2, w // 2] >= 0:
                self.colour_of[i] = int(pat[h // 2, w // 2])
        return i


class AdaptiveEffectModel(EffectModel):
    """EffectModel whose recolour successor table is learned per level; the transferred period is only a guess."""

    def __init__(self, dyn_class, period_guess=None):
        super().__init__(dyn_class, succ=None, period=None)
        self.period_guess = period_guess
        self.guess = {}                              # sig -> sig, from the period guess only

    @property
    def period(self):
        # length of the cycles actually observed closing on this level (if they agree), else the transferred guess
        lengths = set()
        for a in self.succ:
            n, b = 1, self.succ[a]
            while b != a and b in self.succ and n <= len(self.succ):
                b, n = self.succ[b], n + 1
            if b == a:
                lengths.add(n)
        return lengths.pop() if len(lengths) == 1 else self.period_guess

    @period.setter
    def period(self, value):                         # EffectModel.__init__ assigns it; ignore
        pass

    def next_colour(self, sig):
        return self.succ.get(sig, self.guess.get(sig))

    def observe(self, akey, pre, post, sig_of):
        if self.dyn_class != "recolour":
            return super().observe(akey, pre, post, sig_of)
        if pre == post:
            self.noop.add(akey); return
        self.noop.discard(akey)
        changed = frozenset(i for i, (a, b) in enumerate(zip(pre, post)) if a != b)
        for i in changed:
            self.succ[pre[i]] = post[i]              # observed: always wins over a guess
            self.guess.pop(pre[i], None)
            if self.period == 2 and post[i] not in self.succ:
                self.guess[post[i]] = pre[i]
        self.recol[akey] = changed

    def apply(self, compiled, state):
        kind, data = compiled
        if kind == "move":
            return tuple(state[p] for p in data)
        s = list(state)
        for i in data:
            nxt = self.next_colour(s[i])
            if nxt is None:
                return None
            s[i] = nxt
        return tuple(s)


class AdaptivePlannerAgent(KnowledgePlannerAgent):
    name = "E19_adaptive_planner"

    def __init__(self, seed=0, verify_streak=3, verify_cap=12, **kw):
        super().__init__(seed=seed, **kw)
        self.verify_streak = verify_streak
        self.verify_cap = verify_cap

    def _begin_level(self, grid):
        super()._begin_level(grid)
        self.v_streak, self.v_count = defaultdict(int), defaultdict(int)
        self._verifying = None
        self.hyp_i = defaultdict(int)                # LegendTemplate -> index of the grounding hypothesis in use
        self.verify_clicks = getattr(self, "verify_clicks", 0)

    def reset(self):
        self.verify_clicks = 0
        super().reset()
        self.sigs = CentreSigTable()

    def act(self, obs, legal):
        if obs.levels_completed > self.level and self.goal is not None and getattr(self.goal, "mapping", None):
            self.goal_tmpl.mapping = self.goal.mapping   # a hypothesis that cleared the level becomes knowledge
        return super().act(obs, legal)

    def _init_planning(self):
        super()._init_planning()
        # change 1: per-level successor table, transferred period kept only as a guess
        self.effects = AdaptiveEffectModel(self.k2_class, period_guess=self.k2_period)
        # change 4: also probe each object of a kind never seen on an earlier level (after the K1 matches)
        known = set(self.sigs.ids)
        queued = set(self.probe_queue)
        novel = [o.pos for o in self.layout.objs if o.sig not in known and o.pos not in queued]
        self.probe_queue += novel
        self.verify_order = list(dict.fromkeys(self.probe_queue))
        self.log.append(("novel-probes", self.level, len(novel)))

    # -- change 2: active verification of move-class effects --------------------------------------------------------
    def _observe_click(self, grid):
        pend, verifying, mis_before = self.pending, self._verifying, self.mispredictions
        super()._observe_click(grid)
        if pend is not None and verifying is None and self.mispredictions > mis_before:
            akey = pend[0]                           # change 5: re-verify the button that just surprised us
            self.v_streak[akey] = 0
            self.v_count[akey] = min(self.v_count[akey], self.verify_cap - 3)
            return
        if pend is None or verifying is None or pend[0] != verifying or self.layout is None:
            return
        post, _ = self.layout.encode(grid)
        if pend[2] is None or post is None:
            return
        if post == pend[2]:
            self.v_streak[verifying] += 1
        else:
            self.v_streak[verifying] = 0
            self.mispredictions = mis_before         # expected while verifying; not a planning failure

    def _verification_step(self, state):
        if self.effects.dyn_class != "move":
            return None
        compiled = self.effects.compile(self.layout.slots)
        def settled(a):
            # enough correct predictions in a row AND every moved slot has a single possible source
            cands = self.effects.cands.get(a)
            unique = cands is not None and all(len(c) == 1 for q, c in enumerate(cands) if q not in c)
            return self.v_streak[a] >= self.verify_streak and unique

        todo = [a for a in self.verify_order
                if a in compiled and a not in self.effects.noop
                and not settled(a) and self.v_count[a] < self.verify_cap]
        if not todo:
            return None
        akey = min(todo, key=lambda a: self.v_count[a])  # round-robin -> varied states
        self.v_count[akey] += 1
        return akey, self.effects.apply(compiled[akey], state)

    def _plan_act(self, grid, legal):
        self._verifying = None
        state, _ = self.layout.encode(grid)
        if state is None or not any(a.is_complex() for a in legal):
            return super()._plan_act(grid, legal)
        while self.probe_queue:                      # probes first, exactly as E18
            pos = self.probe_queue.pop(0)
            if not self.effects.known(pos):
                return self._click(legal, pos, "probe", grid)
        if self.goal is not None and not self.plan and self.goal.test(state):
            # change 3: the grounded goal says "solved" but the level did not end -> refuted, try the next grounding
            self.log.append(("goal-refuted", self.level, self.goal.name, str(getattr(self.goal, "mapping", ""))))
            if isinstance(self.goal_tmpl, LegendTemplate):
                self.hyp_i[self.goal_tmpl] += 1
            else:
                self.goal_tried.append(self.goal_tmpl)
            self.goal = None
        step = self._verification_step(state)
        if step is not None:
            akey, predicted = step
            self._verifying = akey
            self.verify_clicks += 1
            self.plan = []
            return self._click(legal, akey, "verify", grid, predicted=predicted)
        return super()._plan_act(grid, legal)

    # -- changes 3 + 4: goal grounding ------------------------------------------------------------------------------
    def _goal_layout(self):
        d = self.layout.as_dict(model_dyn=self._model_dynamic())
        effectful = {a for a in list(self.effects.recol) + list(self.effects.cands) if a not in self.effects.noop}
        d["static"] = [i for i in d["static"] if self.layout.slots[i] not in effectful]   # buttons are not legends
        return d

    @staticmethod
    def _legend_hypotheses(tmpl, d):
        seen = {v for _, cells in tmpl._keys(d) for v, _ in cells}
        unknown = sorted(v for v in seen if v not in (tmpl.mapping or {}))
        return [dict(tmpl.mapping or {}, **dict(zip(unknown, combo)))
                for combo in product([("any",), ("diff",), ("same",)], repeat=len(unknown))][:27]

    def _ground_goal(self):
        d = self._goal_layout()
        for tmpl in self.k3:
            if tmpl in self.goal_tried:
                continue
            if isinstance(tmpl, LegendTemplate) and tmpl.mapping:
                hyps = self._legend_hypotheses(tmpl, d)
                while self.hyp_i[tmpl] < len(hyps):
                    g = tmpl.ground(d, hyps[self.hyp_i[tmpl]])
                    if g is not None:
                        g.mapping = hyps[self.hyp_i[tmpl]]
                        return tmpl, g
                    self.hyp_i[tmpl] += 1
                continue
            g = tmpl.ground(d)
            if g is not None:
                return tmpl, g
        return None, None

    # -- change 1: goal solving for recolour cycles of any length ---------------------------------------------------
    def _solve_independent(self, state, compiled):
        """When every click recolours exactly one tile and no tile is shared, each goal condition is satisfied by
        clicking its own tile 0..k-1 times -- valid for any cycle length."""
        acts = {a: c[1] for a, c in compiled.items() if c[0] == "recolour"}
        slot_act = {}
        for a, touched in acts.items():
            if len(touched) != 1 or touched[0] in slot_act:
                return None
            slot_act[touched[0]] = a
        by_slot = defaultdict(list)
        for p, rel, centre in self.goal.conds:
            by_slot[p].append((rel, centre))
        colour_of = self.sigs.colour_of
        plan, s_ = [], state
        for p, conds in by_slot.items():
            c, presses = state[p], 0
            while not all(legend_ok(colour_of[c], rel, centre) for rel, centre in conds):
                if p not in slot_act or presses > 8:
                    return None
                nxt = self.effects.next_colour(c)
                if nxt is None:                      # successor never seen: one exploratory click on this tile
                    return [(slot_act[p], None)]
                c, presses = nxt, presses + 1
                s_ = self.effects.apply(compiled[slot_act[p]], s_)
                plan.append((slot_act[p], s_))
        return plan if plan and self.goal.test(s_) else None

    def _plan_to_goal(self, state, compiled):
        if getattr(self.goal, "conds", None) is not None and self.effects.dyn_class == "recolour":
            plan = self._solve_independent(state, compiled)
            if plan is not None:
                return plan
            # E18's GF(2) solver reads effects.succ directly: give it observed successors plus period guesses
            observed = self.effects.succ
            self.effects.succ = {**self.effects.guess, **observed}
            try:
                return super()._plan_to_goal(state, compiled)
            finally:
                self.effects.succ = observed
        return super()._plan_to_goal(state, compiled)

    def stats(self):
        s = super().stats()
        s["verify_clicks"] = self.verify_clicks
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--games", default=",".join(e18.LIVE_GAMES))
    ap.add_argument("--budget", type=int, default=e18.BUDGET)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    arc = Arcade()
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for g in args.games.split(","):
            sc = arc.create_scorecard(tags=["E19", f"s{seed}"])
            r = e18.run_episode(arc, g, AdaptivePlannerAgent(seed=seed), sc, max_steps=args.budget, seed=seed)
            r["seed"] = seed
            try:
                arc.close_scorecard(sc)
            except Exception:
                pass
            print(f"{r['agent']:22s} s{seed} {g}: levels={r['levels']} clears={r['level_clears_log']} "
                  f"verify={r['verify_clicks']} t={r['seconds']}s", flush=True)
            rows.append(r)
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
