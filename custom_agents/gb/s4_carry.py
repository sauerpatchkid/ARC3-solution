"""s4_carry.py - S4 CARRY: the book's rules re-bound on the next level, and how they do there.

The problem it fixes: a rule's PARAMS hold one level's facts (its colours, its
regions), so a book frozen on one level says little about the next until each
rule is re-bound there (plan section 5.4). Until now only CP1's import scored
transfer, with CP1's own re-bindings, so nothing new could be carried.

Version 0 is CP1's procedure (legacy/rulebook/tools/wm_tier0a_v2.py, rebind and
score_book), nothing more: each rule of the training level's book is fitted to
the transfer level's FIT split by bind.search (its training binding preferred),
bind.choose picks the binding, and the rule is in T1 if that binding is exact
there, the rule is trusted over both levels, and the lint finds no level fact
in its code. T0 (the book untouched) and T1 are scored on the SCORE split, per
move, against "nothing" and "memory", with the gate's interval
(metrics.gate_interval). The search runs to a time budget (bind.BUDGET_S), so a
slow machine could stop it early; `untried` says when it did.

Version 1 adds two things from plan 5.4, each aimed at a loss G1 showed:

  ROLES (`[carry] roles_in_bind`, off by default so S4 still gives CP1's
  re-bindings). When several bindings are exact on the fit split ("binding
  ambiguous") and apply to the same, largest number of its cases, the one
  whose colour slots keep their ROLES wins: a slot bound to the mover on the
  training level should be bound to the mover on the next (S1's roles of each
  level: mover, static, click changers, click no-ops, counters). Remaining ties
  go to the search's own order. G1's m0r0 rules lost T1 this way: 113 exact
  bindings all applied to 22 cases, and the search's first moved `agent` from
  the mover (colour 10) to a static colour (6). The hint only reorders bindings
  the search already found exact, and never prefers one that applies to fewer
  cases: tried first without that limit (18 Oct, dev games, exploratory), it
  cost ft09 3.9 points of T1 by choosing a colour that applied to 2 cases over
  one that applied to 6 (S1's role sets differ by "counter").
  ONLINE WAIT (`ready`, used by S5 online and in its replay). A carried rule is
  re-bound on a new level once its family of actions has `online_min_cases`
  distinct changing cases there, or after `online_max_wait` moves, whichever
  comes first (vc33's thin fit window). Offline gates keep CP1's 300-move fit
  split, so this does not change T1.

Repair calls (plan 5.4, step 5) come with H2.

LATER LEVELS (`later_levels`, 19 Oct; `--stage carry --later`). T1 looks one level
ahead. IMAGINE (plan 5.5) would use the book on every later level, where Goose
stalls, to stand in for moves not yet tried. So for each level after the training
level of a recording: every book rule is re-bound on the level's first
`online_max_wait` moves (S5's online wait, at its longest) by CP1's rebind, the
rules kept are composed on the rest of the level, and each (screen, action) pair
there that the first moves never saw is scored at its first occurrence (on a
level with more pairs than the evidence index keeps, MAX_KEYS, its uniform sample
of them): does the
book apply (claim a cell), is the whole next screen then right (cells it leaves
unclaimed unchanged), is a claimed cell wrong, does it claim every live cell (the
plan's IMAGINE condition). The decoration mask is S0's, kept from level to level
(the training level's and every later level's so far, together): IMAGINE would
compare screens by the map's key, whose mask is sticky across levels. tu93 shows
why: its move-budget bar is masked on level 3, but on level 4 it changes 3 cells
a move, which neither S0's ticker scan nor gridtools.tick_cells (components of at
most 2 cells) catches, so level 4's own mask is empty.

Reads  <run>/evidence/<game>_L<k>_train.npz, <game>_L<j>_{fit,score}.npz
       <run>/s2_write/<game>/candidates.jsonl
       <run>/s3_check/<game>_L<k>.book.json
       <run>/s1_see/<game>_L<k>_train.roles.json, <game>_L<j>_fit.roles.json   (with roles_in_bind)
Writes <run>/s4_carry/<game>_L<j>.carry.json      each rule's re-binding, T0 and T1
       <run>/s4_carry/<game>_L<j>.t1_status.npz   per case of the score split: T1's status
"""
import ast
import glob
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xxhash

from gb.artifacts import plain
from gb.core import bind as B
from gb.core import book as K
from gb.core.evidence import LevelEvidence, read_moves, ticker_mask
from gb.core.metrics import gate_interval, memory_hash, moves_right, nothing_hash
from gb.s0_mask import complete_bars
from gb.s3_check import member, regrade, status_arrays

VERSION = 1                     # bump when what the output means changes (artifacts.py)
WORKERS = 8                     # rules re-bound at the same time (each in its own child process)
DEFAULTS = {"roles_in_bind": False, "online_min_cases": 10, "online_max_wait": 1000}
ROLES = ("static", "click_changers", "click_noops", "counters")   # besides "mover"


def settings_of(run):
    """S4's settings from the run's [carry] section. The header records only what
    differs from CP1's procedure, so runs made before version 1 stay current."""
    got = dict(DEFAULTS, **{k: v for k, v in run.config.get("carry", {}).items() if k in DEFAULTS})
    return got, ({"roles_in_bind": True} if got["roles_in_bind"] else {})


# ---- roles: which exact binding keeps each colour slot's role -----------------------------------------
def colour_slots(src):
    """The names of a rule's Colour() slots, read from its PARAMS without running it."""
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            return [k.value for k, v in zip(node.value.keys, node.value.values)
                    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "Colour"]
    return []


def role_set(roles, colour):
    """A colour's roles on one level (S1's roles.json): a frozenset of names."""
    out = {r for r in ROLES if colour in roles.get(r, [])}
    if colour == roles.get("mover") or colour in roles.get("movers", []):
        out.add("mover")
    return frozenset(out)


def role_score(slots, before, after, roles_before, roles_after):
    """How many colour slots keep their role from `before` (training binding) to `after`."""
    return sum(1 for n in slots if n in before and n in after and before[n] is not None
               and role_set(roles_before, before[n]) == role_set(roles_after, after[n]))


def by_roles(c, found, best, p_fit, roles):
    """Among the exact bindings that apply to the most cases of the fit split, the one
    keeping the most colour roles (the search's order breaks ties). Returns (final, note)."""
    slots = colour_slots(c["code"])
    top_applies = max((e["applies"] for e in found.get("exact_list", [])), default=0)
    exact = [e for e in found.get("exact_list", []) if e["applies"] and e["applies"] == top_applies]
    if not slots or len(exact) < 2:
        return best, None
    score = [role_score(slots, c["binding"], e["binding"], *roles) for e in exact]
    top = max(range(len(exact)), key=lambda i: (score[i], -i))
    note = {"score": score[top], "score_search": role_score(slots, c["binding"], best["binding"], *roles)}
    if exact[top]["binding"] == best["binding"] or note["score"] <= note["score_search"]:
        return best, dict(note, chosen="search")
    res = B.run_rule(c["code"], p_fit, binding=exact[top]["binding"])
    if res.get("stage") != "checked" or not res["claimed_ok"][res["applies"]].all():
        return best, dict(note, chosen="search", why="the role binding did not re-run exact")
    return {"binding": exact[top]["binding"], "exact": True, "guess": False, "res": res}, dict(note, chosen="roles")


# ---- the online wait --------------------------------------------------------------------------------
def ready(moves_on_level, distinct_changing, settings=DEFAULTS):
    """Has a carried rule waited long enough on a new level to be re-bound? Its
    family has `online_min_cases` distinct changing cases there, or the level has
    had `online_max_wait` moves."""
    return distinct_changing >= settings["online_min_cases"] or moves_on_level >= settings["online_max_wait"]


def rebind(c, res_train, train, fit, p_fit, roles=None):
    """One rule re-bound on the fit split (CP1's rebind): {in_t1, flag, why, binding, ...}.
    `roles`: (training level's, fit split's) S1 roles, to break a tie between exact bindings."""
    found = B.search(c["code"], p_fit, family=B.family_of(c["group"]), prefer=c["binding"])
    if found["stage"] != "checked":
        return {"in_t1": False, "flag": "binding failed", "why": found.get("reason"), "binding": None}
    best, flag = B.choose(found)
    out = {"in_t1": False, "flag": flag, "binding": None, "untried": found["untried"]}
    if roles is not None and flag == "binding ambiguous":
        best, out["roles"] = by_roles(c, found, best, p_fit, roles)
    if best is None:
        return dict(out, why="no binding is exact on the fit split")
    t = K.trust([(res_train, train), (best["res"], fit)])
    out.update(binding=best["binding"], trust={k: v for k, v in t.items() if k != "levels"})
    if c["lint"]:
        return dict(out, flag="level-bound rule", why="the lint finds level facts in its code")
    return dict(out, in_t1=t["trusted"], why=None if t["trusted"] else f"not trusted across both levels: {t['reason']}")


def score_book(members, p_score, score, base):
    """A book on the score split: its numbers, per-move rates and interval (CP1's score_book)."""
    comp = K.compose(members, p_score)
    if comp["stage"] != "checked":
        raise RuntimeError(f"the book could not be composed: {comp.get('reason')}")
    nums = K.numbers(comp, score)
    gate = gate_interval(score, dict(base, book=moves_right(score, comp["hash"])))
    return dict(nums, per_move=gate, rules_in=len(members),
                clashing_rules=[i for i, c in enumerate(comp["clashes"]) if c]), comp


def baselines(train, fit, score):
    """Per recorded move of the score split: is "nothing" right, is "memory" right."""
    return {"nothing": moves_right(score, nothing_hash(score)),
            "memory": moves_right(score, memory_hash([train, fit], score))}


def run_game(run, game, log=print, force=False):
    """S4 for one game in `run` (module doc). Returns {t0, t1}, or None if the
    game has no transfer level or the files are current."""
    ev_dir = os.path.join(run.dir, "evidence")
    books = sorted(glob.glob(os.path.join(run.stage_dir("s3_check"), f"{game}_L*.book.json")))
    if not books:
        raise FileNotFoundError(f"{run.name}: no book for {game}; run S3 first")
    with open(books[0]) as f:
        book = json.load(f)
    k = book["level"]
    later = sorted(int(os.path.basename(p)[len(game) + 2:].split("_")[0])
                   for p in glob.glob(os.path.join(ev_dir, f"{game}_L*_fit.npz")))
    j = next((x for x in later if x > k), None)
    p_train = os.path.join(ev_dir, f"{game}_L{k}_train.npz")
    p_fit, p_score = (os.path.join(ev_dir, f"{game}_L{j}_{s}.npz") for s in ("fit", "score"))
    if j is None or not os.path.exists(p_score):
        log(f"[{game}] S4: no transfer level with a score split; nothing to carry")
        return None
    p_cands = os.path.join(run.stage_dir("s2_write"), game, "candidates.jsonl")
    inputs = {"train": p_train, "fit": p_fit, "score": p_score, "candidates": p_cands, "book": books[0]}
    settings, key_settings = settings_of(run)
    roles = None
    if settings["roles_in_bind"]:
        p_roles = [os.path.join(run.dir, "s1_see", f"{game}_L{x}_{s}.roles.json") for x, s in ((k, "train"), (j, "fit"))]
        inputs.update(roles_train=p_roles[0], roles_fit=p_roles[1])
        roles = tuple(json.load(open(p)) for p in p_roles)
    if not force and run.is_current("s4_carry", game, VERSION, key_settings, inputs):
        log(f"[{game}] S4 is current; skipped")
        return None
    with open(p_cands) as f:
        by_id = {c["id"]: c for c in map(json.loads, f)}
    rules = [by_id[i] for i in book["members"]]
    train, fit, score = (LevelEvidence.load(p) for p in (p_train, p_fit, p_score))
    with ThreadPoolExecutor(WORKERS) as ex:
        res_train = list(ex.map(lambda c: regrade(c, p_train, train)[0], rules))
        carried = list(ex.map(lambda cr: rebind(cr[0], cr[1], train, fit, p_fit, roles), zip(rules, res_train)))
    base = baselines(train, fit, score)
    t0, _ = score_book([member(c, p_train) for c in rules], p_score, score, base)
    kept = [(c, t) for c, t in zip(rules, carried) if t["in_t1"]]
    t1, comp1 = score_book([member(c, p_fit, t["binding"]) for c, t in kept], p_score, score, base)
    out = run.stage_dir("s4_carry")
    p_carry, p_t1 = os.path.join(out, f"{game}_L{j}.carry.json"), os.path.join(out, f"{game}_L{j}.t1_status.npz")
    with open(p_carry, "w") as f:
        json.dump(plain({"game": game, "from_level": k, "level": j,
                         "rules": [{"id": c["id"], "group": c["group"], "binding_before": c["binding"], **t}
                                   for c, t in zip(rules, carried)],
                         "t0": t0, "t1": t1, "source": "s4_carry"}), f, indent=1)
    np.savez_compressed(p_t1, **status_arrays(comp1), members=np.array([c["id"] for c, _ in kept]))
    run.write_header("s4_carry", game, VERSION, key_settings, inputs, {"carry": p_carry, "t1_status": p_t1},
                     code=__file__)
    untried = sum(t.get("untried", 0) for t in carried)
    log(f"[{game}] S4: level {k} -> {j}: {len(kept)} of {len(rules)} rules in T1; T1 {t1['per_move']['rate']['book']:.1%} "
        f"(nothing {t1['per_move']['rate']['nothing']:.1%}, memory {t1['per_move']['rate']['memory']:.1%}), "
        f"wrong {t1['wrong']:.1%}" + (f"; WARNING: the search left {untried} bindings untried" if untried else ""))
    return {"t0": t0, "t1": t1}


# ---- every later level of one recording (module doc, LATER LEVELS) ---------------------------------
MIN_SCORE = 200                 # moves a level needs after the wait to be scored


def book_of(book_dir, game):
    """A finished run's book for `game`: (training level, its rules, the training evidence path)."""
    books = glob.glob(os.path.join(book_dir, "s3_check", f"{game}_L*.book.json"))
    if not books:
        raise FileNotFoundError(f"{book_dir}: no book for {game}")
    with open(books[0]) as f:
        book = json.load(f)
    with open(os.path.join(book_dir, "s2_write", game, "candidates.jsonl")) as f:
        by_id = {c["id"]: c for c in map(json.loads, f)}
    k = book["level"]
    return k, [by_id[i] for i in book["members"]], os.path.join(book_dir, "evidence", f"{game}_L{k}_train.npz")


def screen_key(board):
    return xxhash.xxh64(np.ascontiguousarray(board, np.uint8).tobytes()).intdigest()


def pair_keys(frames, actions, live):
    """(screen under the mask, action) per move, as S5's cache and the evidence index key them."""
    return [(screen_key(np.where(live, f, 255)), int(a)) for f, a in zip(frames, actions)]


def later_levels(book_dir, game, corpus, settings=DEFAULTS, log=print):
    """The book re-bound and scored on every level after its training level in one
    recording (module doc, LATER LEVELS). One row of counts per level played."""
    k, rules, p_train = book_of(book_dir, game)
    train = LevelEvidence.load(p_train)
    with ThreadPoolExecutor(WORKERS) as ex:
        res_train = list(ex.map(lambda c: regrade(c, p_train, train)[0], rules))
    wait, rows, level0 = int(settings["online_max_wait"]), [], k      # level0: the level after training, 0-based
    sticky = train.mask.copy()                                         # module doc: the mask kept from level to level
    with tempfile.TemporaryDirectory() as tmp:
        while True:
            f, n, a, order = read_moves(corpus, level0)
            if not len(a):
                break
            own = complete_bars(ticker_mask(f[:20000], n[:20000])[0])
            mask = sticky | own
            sticky = mask
            live = ~mask
            row = dict(level=level0 + 1, moves=int(len(a)), wait=wait, mask_own=int(own.sum()), mask=int(mask.sum()))
            rows.append(row)
            level0 += 1
            if len(a) < wait + MIN_SCORE:
                row["skipped"] = f"fewer than {wait + MIN_SCORE} moves"
                continue
            part = lambda s: None if order is None else {key: v[s] for key, v in order.items()}
            ev_fit, ev_sc = (LevelEvidence.from_moves(game, level0 - 1, f[s], n[s], a[s], mask=mask, order=part(s))
                             for s in (slice(0, wait), slice(wait, None)))
            p_fit, p_sc = os.path.join(tmp, f"L{level0}_fit.npz"), os.path.join(tmp, f"L{level0}_score.npz")
            ev_fit.save(p_fit)
            ev_sc.save(p_sc)
            with ThreadPoolExecutor(WORKERS) as ex:
                carried = list(ex.map(lambda cr: rebind(cr[0], cr[1], train, ev_fit, p_fit) if cr[1] is not None
                                      else {"in_t1": False, "flag": None, "why": "could not be re-graded"},
                                      zip(rules, res_train)))
            kept = [(c, t) for c, t in zip(rules, carried) if t["in_t1"]]
            row.update(kept=[c["id"] for c, _ in kept], untried=sum(t.get("untried", 0) for t in carried),
                       why={c["id"]: t["flag"] or t["why"] for c, t in zip(rules, carried) if not t["in_t1"]})
            seen, new = set(pair_keys(f[:wait], a[:wait], live)), []
            for i, key in enumerate(pair_keys(f[wait:], a[wait:], live)):
                if key not in seen:
                    seen.add(key)
                    new.append(i)
            new = np.array(new, int)
            sampled = ev_sc.moves["key"][new] >= 0          # a level over MAX_KEYS pairs keeps a uniform sample
            new = new[sampled]
            changing = ((f[wait:] != n[wait:]) & live).reshape(len(a) - wait, -1).any(axis=1)[new]
            row.update(new=len(new), new_sampled_out=int((~sampled).sum()), new_changing=int(changing.sum()),
                       nothing_right=int(moves_right(ev_sc, nothing_hash(ev_sc))[new].sum()))
            if not kept:
                continue
            comp = K.compose([member(c, p_fit, t["binding"]) for c, t in kept], p_sc)
            if comp["stage"] != "checked":
                row["error"] = comp.get("reason")
                continue
            case = ev_sc.moves["key"][new]
            right = moves_right(ev_sc, comp["hash"])[new]
            claimed = np.asarray(comp["claimed_n"])[case]
            applies, full = claimed > 0, claimed >= int(live.sum())
            wrong = np.asarray(comp["status"])[case] == K.WRONG
            for name, sel in (("", np.ones(len(new), bool)), ("_changing", changing)):
                row.update({f"applies{name}": int((applies & sel).sum()),
                            f"right_applies{name}": int((applies & right & sel).sum()),
                            f"wrong{name}": int((wrong & sel).sum()),
                            f"full_right{name}": int((full & right & sel).sum())})
            log(f"[{game}] level {row['level']}: {len(kept)} of {len(rules)} rules re-bound; {row['new_changing']} "
                f"new changing pairs, the book applies to {row['applies_changing']}, right on "
                f"{row['right_applies_changing']}, wrong on {row['wrong_changing']}")
    return rows


def later_table(per_game, log=print):
    """Rows of every seed pooled per game and level: markdown text."""
    pct = lambda x, y: f"{x / y:.1%}" if y else "-"
    lines = ["| Game | Level | Seeds | Rules re-bound | New pairs (changing) | Book applies (changing) | "
             "Right where it applies (changing) | Wrong (changing) | Claims every cell, right (changing) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for game, rows in per_game.items():
        for level in sorted({r["level"] for r in rows}):
            rs = [r for r in rows if r["level"] == level and "new" in r]
            if not rs:
                continue
            s = lambda key: sum(r.get(key, 0) for r in rs)
            nc, ac = s("new_changing"), s("applies_changing")
            lines.append(f"| {game} | {level} | {', '.join(str(r['seed']) for r in rs)} | "
                         f"{', '.join(str(len(r.get('kept', []))) for r in rs)} | {s('new'):,} ({nc:,}) | "
                         f"{pct(s('applies'), s('new'))} ({pct(ac, nc)}) | "
                         f"{pct(s('right_applies'), s('applies'))} ({pct(s('right_applies_changing'), ac)}) | "
                         f"{pct(s('wrong'), s('applies'))} ({pct(s('wrong_changing'), ac)}) | "
                         f"{pct(s('full_right'), s('new'))} ({pct(s('full_right_changing'), nc)}) |")
    text = "\n".join(lines) + "\n"
    log(text)
    return text
