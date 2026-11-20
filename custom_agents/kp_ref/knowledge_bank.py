"""A persistent knowledge bank shared across levels, games and episodes (E23).

Everything in the bank was written by the agent from its own play. Nothing is given to it: it is not told game names
(a game is recognised by a fingerprint of the first frame it sees), rules, or mechanics.

  games      per game fingerprint: the full learned store of the last episode on that game
             (click games: E20's goal rules, symbol meanings, interactive kinds, effect family;
              keyboard games: E22's action effects, controlled kind, movement model, clock lines, goal rules)
  families   abstract goal-rule families with the game-specific shape removed, e.g. "AA|frame|inside|same",
             with the number of games that confirmed / refuted them
  effects    for click games, how often each effect family (recolour / move) was found
  actions    per keyboard action name: in how many games it moved the controlled object, moved nothing,
             or was the non-moving action that ended a level

Use at the start of an episode:
  same game seen before  -> restore its store and plan from level 1 (knowledge is still re-tested as usual)
  new game               -> abstract knowledge only, as unconfirmed priors:
                            click games: hypotheses of confirmed families are grounded on this game's objects, and
                            the most common effect family is the starting guess (E19 re-verifies it);
                            keyboard games: actions that ended levels elsewhere without moving anything get the
                            level-ending bonus
"""
import json, os
from collections import Counter, defaultdict

from e20_learned_knowledge import Hyp


def family_of(key):
    q, subj, _shape, rel, crel = key.split("|")
    return f"{q}|{subj}|{rel}|{crel}"


def hyp_from_key(key):
    q, subj, shape, rel, crel = key.split("|")
    return Hyp(q, subj, None if shape == "None" else shape, rel, None if crel == "None" else crel)


# -- (de)serialising the agents' stores ---------------------------------------------------------------------------------
def _shape_out(s):
    (h, w), b = s
    return [int(h), int(w), b.hex()]


def _shape_in(v):
    return ((v[0], v[1]), bytes.fromhex(v[2]))


def _mapping_out(m):
    return None if not m else {str(k): list(v) for k, v in m.items()}


def _mapping_in(m):
    return None if not m else {int(k): tuple(v) for k, v in m.items()}


def click_store_out(agent):
    st = agent.store
    return dict(
        k3={k: {**{kk: v for kk, v in rec.items() if kk != "mapping"}, "mapping": _mapping_out(rec.get("mapping"))}
            for k, rec in st.stats.items()},
        k1=st.k1, k1_shapes=[_shape_out(s) for s in agent.k1_shapes], k1_sigs=[_shape_out(s) for s in agent.k1_sigs],
        k2_class=agent.k2_class, k2_period=agent.k2_period, k2=st.k2)


def click_store_in(agent, d):
    st = agent.store
    for k, rec in d["k3"].items():
        st.hyps[k] = hyp_from_key(k)
        st.stats[k] = {**rec, "mapping": _mapping_in(rec.get("mapping")), "levels": []}
    st.k1 = dict(d["k1"])
    agent.k1_shapes = {_shape_in(v) for v in d["k1_shapes"]}
    agent.k1_sigs = {_shape_in(v) for v in d["k1_sigs"]}
    agent.k2_class, agent.k2_period = d["k2_class"], d["k2_period"]
    agent.k3 = st.ranked()


def keyboard_store_out(xk):
    return dict(action_stats=xk.action_stats, mover=dict(xk.mover), disp={a: dict(c) for a, c in xk.disp.items()},
                clock=[list(l) for l in xk.clock], goal=xk.goal, consumed=xk.consumed, trigger=dict(xk.trigger),
                blocked=dict(xk.blocked), passable=dict(xk.passable), consume=dict(xk.consume),
                rules={k: {**v, "rule": [list(a) for a in v["rule"]], "levels": []} for k, v in xk.rules.items()})


def keyboard_store_in(xk, d):
    xk.action_stats = {k: list(v) for k, v in d["action_stats"].items()}
    xk.mover = Counter(d["mover"])
    xk.disp = defaultdict(Counter, {a: Counter(c) for a, c in d["disp"].items()})
    xk.clock = [tuple(l) for l in d["clock"]]
    xk.goal, xk.consumed, xk.trigger = d["goal"], d["consumed"], Counter(d["trigger"])
    xk.blocked, xk.passable, xk.consume = Counter(d["blocked"]), Counter(d["passable"]), Counter(d["consume"])
    xk.rules = {k: {**v, "rule": tuple(tuple(a) for a in v["rule"])} for k, v in d["rules"].items()}


# -- the bank -------------------------------------------------------------------------------------------------------------
class KnowledgeBank:
    def __init__(self, path):
        self.path = path
        self.data = dict(games={}, families={}, effects={}, actions={}, log=[])
        if path and os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, default=str)
        os.replace(tmp, self.path)

    def game(self, fp):
        return self.data["games"].get(fp)

    # -- writing --------------------------------------------------------------------------------------------------------
    def record_game(self, fp, kind, levels, store, episode_tag):
        g = self.data["games"].setdefault(fp, dict(kind=kind, episodes=[], best_levels=0))
        g["store"] = store
        if episode_tag not in g["episodes"]:
            g["episodes"].append(episode_tag)
        g["best_levels"] = max(g["best_levels"], levels)
        self._rebuild_abstract()
        self.save()

    def _rebuild_abstract(self):
        """Abstract knowledge is recomputed from the per-game stores, so each game counts once."""
        fams, effects, actions = {}, Counter(), {}
        for fp, g in self.data["games"].items():
            st = g.get("store") or {}
            if g["kind"] == "click":
                per = {}
                for key, rec in st.get("k3", {}).items():
                    real = rec["confirmed"] - (1 if rec.get("prior") else 0)
                    f = family_of(key)
                    if real > 0 and rec["refuted"] == 0:
                        per[f] = "confirmed"
                    elif rec["refuted"] > 0 and per.get(f) != "confirmed":
                        per[f] = "refuted"
                for f, v in per.items():
                    e = fams.setdefault(f, dict(games_confirmed=0, games_refuted=0))
                    e["games_" + v] += 1
                if st.get("k2_class"):
                    effects[st["k2_class"]] += 1
            else:
                disp = st.get("disp", {})
                moved = set()                       # same test as the agent's (E23's BankKnowledge.displacement)
                for a, c in disp.items():
                    if not c:
                        continue
                    (vec, n), = Counter(c).most_common(1)
                    rec = actions.setdefault(a, dict(moved_in_games=0, ended_level_without_moving=0, directions={}))
                    changed = st.get("action_stats", {}).get(a, [0, 0])[1]
                    if n >= 2 and vec != "0,0" and n >= 0.3 * max(changed, 1):
                        moved.add(a)
                        rec["moved_in_games"] += 1
                        dy, dx = map(int, vec.split(","))
                        d = f"{(dy > 0) - (dy < 0)},{(dx > 0) - (dx < 0)}"
                        rec["directions"][d] = rec["directions"].get(d, 0) + 1
                for a in st.get("trigger", {}):
                    if a.startswith("ACTION") and a not in moved:
                        rec = actions.setdefault(a, dict(moved_in_games=0, ended_level_without_moving=0,
                                                         directions={}))
                        rec["ended_level_without_moving"] += 1
        self.data["families"], self.data["effects"], self.data["actions"] = fams, dict(effects), actions

    def note(self, msg):
        self.data["log"].append(msg)
        self.data["log"] = self.data["log"][-200:]

    # -- reading --------------------------------------------------------------------------------------------------------
    def family_prior(self):
        """Families confirmed in more games than they were refuted in, best first."""
        f = self.data["families"]
        ok = [k for k, v in f.items() if v["games_confirmed"] > v["games_refuted"]]
        return sorted(ok, key=lambda k: (-(f[k]["games_confirmed"] - f[k]["games_refuted"]), k))

    def effect_prior(self):
        e = self.data["effects"]
        return max(e, key=e.get) if e else None

    def ending_actions(self):
        return [a for a, v in self.data["actions"].items() if v["ended_level_without_moving"] > 0]
