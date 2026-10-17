"""s3_guards.py - S3's guard fitter: learn where a near-miss rule is wrong, and split it in two.

The problem it fixes: CP1 left rules that are right on almost every move but
never exact (m0r0's best: 99.8% of its moves), and Rulebook v2 could only make
such a rule exact by asking the LLM again. A rule that moves the mover left is
typically wrong exactly where something stands on the mover's left, and that is
one of S1's facts. So the condition can be found by search, with no LLM.
(Plan section 5.3, step 2; the design was settled on 16 Oct, in CLAUDE.md.)

For each rule that is admitted but not trusted on its level:

  1. RIGHT AND WRONG. Of the distinct cases the rule applies to, R are those
     where every claimed cell is right and W those where one is not. With fewer
     than `guard_min_wrong` W cases no guard is fitted (too easy to fit by chance).
  2. HOLD-OUT. W and R are each put in case order and shuffled with one
     random.Random(guard_seed), W first; the first `guard_holdout` share of
     each, rounded half up, is held out and the rest is the fit set. So the
     hold-out has wrong cases too (1 of 3, 3 of 10).
  3. SEARCH. A literal is one of S1's facts (h1_vocab.feature_names) or its
     negation. A guard is one literal or, with `guard_max_literals = 2`, two
     joined by AND. It FITS if it is true on every fit W case and false on every
     fit R case. The guards that fit are tried in this order: fewer literals;
     geometry (touch, ray, edge, clear, inside, sits on the background) before
     the subject and the action (size, large, which button) before colour (for
     a pair the worse literal decides, then the two together); then S1's order
     of the facts, a fact before its negation. The first that also separates
     the held-out cases is the guard. Colour literals need
     `guard_colour_features = "params"` ("off": geometry and subject only).
     Directions are the screen's: a rule belongs to one action group, so a
     button rule's direction of travel is fixed and needs no re-aligning.
  4. TWO NEW RULES, as source that wraps the original. Its top-level names that
     the new code defines (applies, predict, bind, RULE, KIND, PRIORITY, guard,
     features) are renamed base_<name> with ast, everywhere they are used; the
     rest of its code, comments included, is kept as written. Then:
       narrowed   applies = base AND NOT guard; predicts what the base predicts
       exception  applies = base AND guard; predicts that every cell the base
                  claims stays as it is (KIND = "exception", PRIORITY = 1).
                  Written with `auto_exception`. It is exact on its level exactly
                  when the W cases' claimed cells did not change (the typical
                  blocked move), which the replay checks.
     The guard calls h1_vocab.features, the sandbox's copy of S1's facts. The
     mover colour (when the guard needs the subject of a button) and every
     colour a guard names are PARAMS slots, so they re-bind on the next level:
     the original's own Colour() slot if it is bound to that colour on this
     level (the guard and the rule then re-bind together, and a 300-move fit
     split that never shows the guard firing cannot pull them apart), else a
     new one, Colour(), whose value here is bind()'s guess. A guard that would
     take the rule over contract.MAX_SLOTS is skipped. A v1 rule comes out in
     v2 form.
  5. REPLAY. Each new rule is graded on every case of the level like any
     candidate (gb/core: bind.run_rule, book.trust), with the original's binding
     plus the new slots' values. A guard is only a hypothesis until the replay
     says otherwise. The narrowed rule is exact on its own level by construction
     (S1 and the sandbox compute the same facts): the hold-out and the next
     level are what test it.

For the game, the composed book on the training level and T1 on the next level
are worked out before and after, the way CP1 did them (cp1_import.py): before =
the trusted rules; after = the same plus every narrowed rule the replay trusts.
"Before" must equal what the import recorded (the book's numbers in
s3_check/<game>_L<k>.book.json, T1 in s4_carry/<game>_L<k+1>.carry.json, both
equal to CP1's registered ones); the summary says whether it does
(`before_matches_import`), and a mismatch is logged as one.
A rule that was in the book keeps its own T1 binding; a new one is re-bound on
the fit split by bind.search, as CP1 re-bound its rules (trusted on both levels,
no lint warning). An exception claims no changed cell, so it never passes v2's
admission bar (gain > 0) and adds nothing to coverage or T1. It is reported on
its own (exact on >= 20 moves with no conflicting case, v1's "known no-op" bar)
for S5's pruning, through S3's tiers when they are built.

Run on CP1's candidates, this IS G0 (pre-registration section 3), so run_game
refuses CP1's candidates unless given g0=True (make gb-guards ... G0=1): that is
for after section 3 is registered.

Reads  <run>/evidence/<game>_L<k>_train.npz, <game>_L<k+1>_{fit,score}.npz
       <run>/s1_see/<game>_L<k>_train.facts.npz and .roles.json
       <run>/s2_write/<game>/candidates.jsonl       code, group, order, binding, origin, lint, t1
       <run>/s3_check/<game>/rules.jsonl            each rule's grade on level k (v2_admitted, v2_trusted)
       <run>/s3_check/<game>/rule_cases.npz         per rule and case: applies, claimed_ok
       <run>/s3_check/<game>_L<k>.book.json         the order the groups were called in
Writes <run>/s3_check/<game>/guards.jsonl           per rule looked at: R, W, the guard or why none, in words
       <run>/s3_check/<game>/guarded.jsonl          the new rules: source, binding, grade, T1
       <run>/s3_check/<game>_L<k>.guards.json       the game: rescues, the book and T1 before and after
"""
import ast
import collections
import glob
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gb import h1_vocab as H
from gb import s1_see
from gb.core import bind as B
from gb.core import book as K
from gb.core import contract as C
from gb.core.check import ADMIT
from gb.core.evidence import LevelEvidence
from gb.core.lint import lint
from gb.core.metrics import memory_hash, moves_right, nothing_hash
from gb.cp1_import import member, plain, score_book

VERSION = 1                     # bump when what the output means changes (artifacts.py)
DEFAULTS = {"guard_max_literals": 2, "guard_holdout": 0.30, "guard_min_wrong": 3,
            "guard_colour_features": "params", "auto_exception": True, "guard_seed": 0}
WORKERS = 8                     # rules replayed at the same time (each in its own child process)
OWN_NAMES = ("applies", "predict", "bind", "RULE", "KIND", "PRIORITY", "guard", "features")
MOVER_SLOT, COLOUR_SLOT = "guard_mover", "guard_colour"
SIDE = {"up": "above it", "down": "below it", "left": "on its left", "right": "on its right"}
EDGE = {"up": "top", "down": "bottom", "left": "left", "right": "right"}
G0_REFUSAL = ("{game}: these are CP1's candidates, and running the guard fitter on them IS G0 "
              "(docs/3-glassbox/glassbox-prereg.md section 3). Register that section first, then pass "
              "--g0 (make gb-guards ... G0=1).")


class GuardError(ValueError):
    """A guard that cannot be written into this rule; str(e) says why."""


# ---- 1-3. the search: feature bits in, a guard out -------------------------------------------------
def category(name):
    """Step 3's order: 0 geometry, 1 the subject and the action, 2 colour."""
    if "colour=" in name:
        return 2
    if name.split("_")[0] in H.DIRS or name in ("subject_inside", "subject_on_background"):
        return 0
    return 1


def needs_subject(name):
    """Does this fact depend on which object the move is about?"""
    return not (name.startswith("action") or name == "clicked_background")


def split(right, wrong, share, seed):
    """Step 2: (fit R, fit W, held-out R, held-out W), as case numbers."""
    rng, parts = random.Random(seed), {}
    for side, cases in (("W", wrong), ("R", right)):
        cases = sorted(int(k) for k in cases)
        rng.shuffle(cases)
        n = int(share * len(cases) + 0.5)
        parts[side] = (cases[n:], cases[:n])
    return parts["R"][0], parts["W"][0], parts["R"][1], parts["W"][1]


def fit_guard(bits, names, right, wrong, settings, max_colours=2):
    """Steps 1-3 for one rule. `bits`: S1's facts, one bool row per case of the
    level, columns in the order of `names`; `right`, `wrong`: case numbers.
    Returns {guard: [(fact, positive)] or None, why: the reason when None, and
    the counts}."""
    out = {"right": len(right), "wrong": len(wrong), "guard": None}
    if len(wrong) < settings["guard_min_wrong"]:
        return dict(out, why=f"only {len(wrong)} wrong cases (a guard needs {settings['guard_min_wrong']})")
    fr, fw, hr, hw = (np.array(x, np.int64) for x in split(right, wrong, settings["guard_holdout"],
                                                          settings["guard_seed"]))
    out.update(fit_right=len(fr), fit_wrong=len(fw), held_right=len(hr), held_wrong=len(hw))
    colours = settings["guard_colour_features"] == "params"
    lits = [(n, pos) for n in names if colours or category(n) < 2 for pos in (True, False)]
    rank = {lit: i for i, lit in enumerate(lits)}
    col = {n: i for i, n in enumerate(names)}
    on = lambda lit, cases: bits[cases, col[lit[0]]] == lit[1]          # the literal on these cases
    usable = [lit for lit in lits if on(lit, fw).all()]                 # true on every fit W case
    fits = [(lit,) for lit in usable if not on(lit, fr).any()]
    if settings["guard_max_literals"] >= 2 and len(usable) > 1:
        hits = np.array([on(lit, fr) for lit in usable], np.int32)       # literal x fit R
        both = hits @ hits.T                                              # fit R cases two literals share
        fits += [(usable[i], usable[j]) for i in range(len(usable)) for j in range(i + 1, len(usable))
                 if both[i, j] == 0 and usable[i][0] != usable[j][0]]
    fits = [g for g in fits if sum(category(n) == 2 for n, _ in g) <= max_colours]
    fits.sort(key=lambda g: (len(g), max(category(n) for n, _ in g), sum(category(n) for n, _ in g),
                             [rank[lit] for lit in g]))
    out.update(fitting=len(fits), failed_holdout=0)
    for g in fits:
        held_w = np.logical_and.reduce([on(lit, hw) for lit in g]) if len(hw) else np.ones(0, bool)
        held_r = np.logical_and.reduce([on(lit, hr) for lit in g]) if len(hr) else np.zeros(0, bool)
        if held_w.all() and not held_r.any():
            return dict(out, guard=[list(lit) for lit in g])
        out["failed_holdout"] += 1
    return dict(out, why="no guard separates the fit cases" if not fits else
                f"all {len(fits)} guards that separate the fit cases fail on the held-out cases")


# ---- the guard in words ------------------------------------------------------------------------------
def fact_words(name, who, colour=None):
    """One of S1's facts in plain words. `who`: what the subject is called;
    `colour`: the text for a colour slot (default: the colour in the name)."""
    d, _, rest = name.partition("_")
    if d in H.DIRS:
        if rest == "touch":
            return f"{who} is touching something {SIDE[d]}"
        if rest == "edge":
            return f"{who} is at the {EDGE[d]} edge of the screen"
        if rest == "clear":
            return f"nothing stands {SIDE[d]} up to the screen edge"
        if rest.startswith("ray<="):
            return f"something stands within {rest[5:]} cells {SIDE[d]}"
        return f"the first thing {SIDE[d]} is colour {colour or rest[7:]}"
    if name.startswith("action"):
        return "the move is a click" if name == "action6" else f"the move is ACTION{name[6:]}"
    if name.startswith("subject_size<="):
        return f"{who} has at most {name[14:]} cells"
    if name.startswith("subject_colour="):
        return f"{who} is colour {colour or name[15:]}"
    return {"subject": f"there is {who} (a mover on the screen, or an object under the click)",
            "subject_large": f"{who} is large (over {H.LARGE_CELLS} cells)",
            "subject_inside": f"{who} is inside a larger object",
            "subject_on_background": f"{who} sits on the background",
            "clicked_background": "the click is on the background"}[name]


def guard_words(guard, who, slots=None):
    """A guard in plain words; with `slots` ({fact: slot name}), a colour reads as its slot."""
    slots = slots or {}
    parts = []
    for name, pos in guard:
        text = fact_words(name, who, f"<{slots[name]}>" if name in slots else None)
        parts.append(text if pos else f"not ({text})")
    return " and ".join(parts)


# ---- 4. the guard as code ------------------------------------------------------------------------------
def top_names(tree):
    """The names a rule's source defines at the top level."""
    out = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for n in tree.body:
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        out |= {t.id for t in targets if isinstance(t, ast.Name)}
    return out


def _params_node(tree):
    for n in tree.body:
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in targets):
            return n.value
    return None


def guard_expr(guard, slots):
    """The guard's test on `f` (the set features() returns), one literal at a time."""
    parts = []
    for name, pos in guard:
        if name in slots:                                   # a colour: its value comes from PARAMS
            fact = 'f"' + name.split("=")[0] + "={p['" + slots[name] + "']}" + '"'
        else:
            fact = json.dumps(name)
        parts.append(f"{fact} {'in' if pos else 'not in'} f")
    return " and ".join(parts)


def wrap(src, kind, guard, slots, values, text, words, parent):
    """Step 4: the new rule (`kind`: "narrowed" or "exception") as source around
    `src`. `slots`: {colour fact: the slot that holds it}, and under MOVER_SLOT
    the slot that holds the mover when the guard needs it (the original's own
    slots or new ones); `values`: the new slots' values on this level (bind's
    guess); `text`: the new RULE; `words`: the guard in words, for its
    docstring. Raises GuardError if it cannot be written."""
    tree = ast.parse(src)
    version, params = C.read_params(src)
    defined = top_names(tree)
    rename = {n: "base_" + n for n in OWN_NAMES if n in defined}
    used = defined | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    clash = sorted((set(rename.values()) & used) | (set(values) & set(params)))
    if clash:
        raise GuardError(f"the rule already uses the name{'s' * (len(clash) > 1)} {', '.join(clash)}")
    # -- edits to the original, by byte offset (ast's columns are UTF-8 bytes) -------------------------
    data = src.encode()
    starts = [0]
    for line in data.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    at = lambda ln, col: starts[ln - 1] + col
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in rename:
            m = re.compile(rb"def\s+").match(data, at(node.lineno, node.col_offset))
            edits.append((m.end(), m.end() + len(node.name.encode()), rename[node.name]))
        elif isinstance(node, ast.Name) and node.id in rename:
            edits.append((at(node.lineno, node.col_offset), at(node.end_lineno, node.end_col_offset),
                          rename[node.id]))
    new_slots = ", ".join(f'"{s}": Colour()' for s in values)
    pnode, params_line = _params_node(tree), None
    if values and pnode is not None and pnode.keys:          # add the new slots after the last one
        last = pnode.values[-1]
        p = at(last.end_lineno, last.end_col_offset)
        edits.append((p, p, ", " + new_slots))
    elif values and pnode is not None:                       # PARAMS = {}
        edits.append((at(pnode.lineno, pnode.col_offset), at(pnode.end_lineno, pnode.end_col_offset),
                      "{" + new_slots + "}"))
    elif values:
        params_line = "PARAMS = {" + new_slots + "}"
    for a, b, new in sorted(edits, reverse=True):
        data = data[:a] + new.encode() + data[b:]
    # -- the new code ---------------------------------------------------------------------------------
    call = "board, act, api, p" if version == 2 else "board, act, api"
    base = lambda fn: f"{rename[fn]}({call})"
    mover = f'p["{slots[MOVER_SLOT]}"]' if MOVER_SLOT in slots else "None"
    inside = ", with_inside=True" if any(n == "subject_inside" for n, _ in guard) else ""
    colour_slots = {k: v for k, v in slots.items() if k != MOVER_SLOT}
    what = {"narrowed": "base AND NOT guard", "exception": "base AND guard: the base's claimed cells stay"}[kind]
    block = [f"# ---- written by S3's guard fitter (gb/s3_guards.py) from rule {parent}: {what} ----",
             f"RULE = {json.dumps(text)}", f'KIND = "{kind}"'] + (["PRIORITY = 1"] if kind == "exception" else [])
    block += [params_line] if params_line else []
    block += ["", "",
              "def guard(board, act, api, p):",
              f'    """True where rule {parent} was wrong on its level: {words}."""',
              f"    f = features(board, act, api, {mover}{inside})",
              f"    return {guard_expr(guard, colour_slots)}"]
    if "bind" in rename or values:
        guess = "{" + ", ".join(f'"{s}": {v}' for s, v in values.items()) + "}"
        block += ["", "", "def bind(board0, api):"]
        if "bind" in rename:
            block += ["    guess = base_bind(board0, api)",
                      "    guess = dict(guess) if isinstance(guess, dict) else {}"]
            block += [f"    guess.update({guess})"] if values else []
            block += ["    return guess"]
        else:
            block += [f"    return {guess}"]
    block += ["", "", "def applies(board, act, api, p):",
              f"    return {base('applies')} and {'not ' * (kind == 'narrowed')}guard(board, act, api, p)",
              "", "", "def predict(board, act, api, p):"]
    if kind == "narrowed":
        block += [f"    return {base('predict')}"]
    else:
        block += [f"    out = {base('predict')}",
                  "    claimed = out[1] if isinstance(out, tuple) else ~api.ticker",
                  "    return np.array(board), claimed"]
    new_src = data.decode().rstrip() + "\n\n\n" + "\n".join(block) + "\n"
    ok, why, _, _ = C.check_form(new_src)
    if not ok:
        raise GuardError(f"the wrapped rule does not pass the contract: {why}")
    return new_src


def rule_text(src):
    """A rule's RULE string, if it is a literal."""
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RULE" for t in n.targets):
            return n.value.value if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) else ""
    return ""


# ---- 1-4 for one rule ------------------------------------------------------------------------------------
def fit_rule(c, applies, claimed_ok, actions, bits, names, mover, settings):
    """Steps 1-4 for one candidate `c` (a candidates.jsonl record), from its
    per-case results on the level and S1's facts there; `mover`: the level's
    mover colour (S1's roles) or None. Returns (record, new candidates)."""
    right, wrong = np.flatnonzero(applies & claimed_ok), np.flatnonzero(applies & ~claimed_ok)
    rec = {"id": c["id"], "group": c["group"], "rule": rule_text(c["code"])}
    _, params = C.read_params(c["code"])
    buttons, clicks = bool((actions[applies] < 5).any()), bool((actions[applies] >= 5).any())
    who = "the mover" if not clicks else "the clicked object" if not buttons else "the subject"
    room = C.MAX_SLOTS - len(params) - 1                       # one slot is kept for the mover
    fit = fit_guard(bits, names, right, wrong, settings, max_colours=max(room, 0))
    rec.update(fit)
    if fit["guard"] is None:
        return rec, []
    guard = [tuple(lit) for lit in fit["guard"]]
    own = {n: v for n, v in (c.get("binding") or {}).items() if isinstance(params.get(n), C.Colour)}
    slots, values = {}, {}

    def slot_for(colour, new):            # the original's slot bound to `colour` here, else a new one
        same = [n for n, v in own.items() if v == colour]
        if same:
            return same[0]
        values[new] = int(colour)
        return new
    if buttons and mover is not None and any(needs_subject(n) for n, _ in guard):
        slots[MOVER_SLOT] = slot_for(mover, MOVER_SLOT)
    for name, _ in guard:
        if category(name) == 2:
            slots[name] = slot_for(int(name.split("=")[1]),
                                   f"{COLOUR_SLOT}{sum(n != MOVER_SLOT for n in values) + 1}")
    colour_slots = {n: s for n, s in slots.items() if n != MOVER_SLOT}
    words = guard_words(guard, who)                              # this level's colours: for the report
    code_words = guard_words(guard, who, colour_slots)           # the slots: for the rule's own text
    rec.update(words=words, slots=slots)
    base_text = rec["rule"] or f"rule {c['id']}"
    texts = {"narrowed": f"{base_text} -- except when {code_words}",
             "exception": f"When {code_words}, nothing the base rule claims changes "
                          f"(the exception to: {base_text})"}
    made = []
    for kind in ("narrowed", "exception") if settings["auto_exception"] else ("narrowed",):
        try:
            code = wrap(c["code"], kind, guard, slots, values, texts[kind], code_words, c["id"])
        except (GuardError, C.RuleError) as e:
            rec.update(guard=None, why=str(e))
            return rec, []
        made.append({"id": f"{c['id']}.{kind}", "parent": c["id"], "kind": kind, "group": c["group"],
                     "origin": c.get("origin"), "duplicate": c.get("duplicate"), "order": c.get("order", 0),
                     "code": code, "binding": dict(c.get("binding") or {}, **values), "guard": fit["guard"],
                     "words": words, "lint": lint(code), "source": "s3_guards"})
    rec["new"] = [m["id"] for m in made]
    return rec, made


# ---- 5. the replay, and the next level --------------------------------------------------------------------
def replay(c, p_train, train):
    """One new rule on every case of its level: (per-case results, trust) or (None, reason)."""
    res = B.run_rule(c["code"], p_train, binding=c["binding"])
    if res.get("stage") != "checked":
        return None, res.get("reason")
    return res, K.trust([(res, train)])


def noop_exact(t):
    """An exception's bar (v1's known no-op): exact everywhere on >= 20 moves, no conflicting case."""
    return bool(t and t["exact_everywhere"] and t["moves"] >= ADMIT["min_moves"] and not t["conflict_cases"])


def carry(c, res_train, train, fit, p_fit):
    """A new rule re-bound on the next level's fit split, as CP1 re-bound its
    rules (legacy/rulebook/tools/wm_tier0a_v2.py, rebind): in T1 if a binding is
    exact there, the rule is trusted over both levels, and lint finds nothing."""
    found = B.search(c["code"], p_fit, family=B.family_of(c["group"]), prefer=c["binding"])
    if found["stage"] != "checked":
        return {"in_t1": False, "flag": "binding failed", "why": found.get("reason"), "binding": None}
    best, flag = B.choose(found)
    out = {"in_t1": False, "flag": flag, "binding": None, "untried": found["untried"]}
    if best is None:
        return dict(out, why="no binding is exact on the fit split")
    t = K.trust([(res_train, train), (best["res"], fit)])
    out.update(binding=best["binding"], trust={k: v for k, v in t.items() if k != "levels"})
    if c["lint"]:
        return dict(out, flag="level-bound rule", why="the lint finds level facts in its code")
    return dict(out, in_t1=t["trusted"], why=None if t["trusted"] else f"not trusted across both levels: {t['reason']}")


# ---- one game -----------------------------------------------------------------------------------------------
def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_game(run, game, g0=False, log=print, force=False):
    """The guard fitter on every admitted-but-not-trusted rule of one game in
    `run`, then the book and T1 before and after (module doc). Returns the
    game's summary, or None if its files are current."""
    settings = dict(DEFAULTS, **{k: v for k, v in run.config.get("check", {}).items() if k in DEFAULTS})
    p_cands = os.path.join(run.stage_dir("s2_write"), game, "candidates.jsonl")
    cands = _jsonl(p_cands)
    if any(c.get("source") == "cp1" for c in cands) and not g0:
        raise RuntimeError(G0_REFUSAL.format(game=game))
    ev_dir, s3 = os.path.join(run.dir, "evidence"), run.stage_dir("s3_check")
    books = sorted(glob.glob(os.path.join(s3, f"{game}_L*.book.json")))
    if not books:
        raise FileNotFoundError(f"{run.name}: no graded book for {game} in s3_check/")
    with open(books[0]) as f:
        book_before = json.load(f)
    k = book_before["level"]
    p_train = os.path.join(ev_dir, f"{game}_L{k}_train.npz")
    p_fit, p_score = (os.path.join(ev_dir, f"{game}_L{k + 1}_{s}.npz") for s in ("fit", "score"))
    split_name = f"{game}_L{k}_train"
    s1 = run.require("s1_see", split_name, s1_see.VERSION)
    if (s1["settings"]["large_cells"], s1["settings"]["ray_max"]) != (H.LARGE_CELLS, H.RAY_MAX):
        raise ValueError(f"{split_name}: S1 ran with large_cells/ray_max other than h1_vocab's, so the "
                         "sandbox could not compute the facts guards are fitted on")
    p_facts = os.path.join(run.dir, "s1_see", split_name + ".facts.npz")
    p_roles = os.path.join(run.dir, "s1_see", split_name + ".roles.json")
    p_rules, p_cases = (os.path.join(s3, game, f) for f in ("rules.jsonl", "rule_cases.npz"))
    p_carry = os.path.join(run.dir, "s4_carry", f"{game}_L{k + 1}.carry.json")
    inputs = {"evidence": p_train, "facts": p_facts, "roles": p_roles, "candidates": p_cands,
              "rules": p_rules, "rule_cases": p_cases, "book": books[0],
              **{s: p for s, p in (("fit", p_fit), ("score", p_score), ("carry", p_carry)) if os.path.exists(p)}}
    name = f"{game}.guards"
    if not force and run.is_current("s3_check", name, VERSION, settings, inputs):
        log(f"[{game}] the guard fitter is current; skipped")
        return None
    train = LevelEvidence.load(p_train)
    facts = np.load(p_facts)
    names = [str(x) for x in facts["feature_names"]]
    bits = np.unpackbits(facts["features"], axis=1)[:, :len(names)].astype(bool)
    with open(p_roles) as f:
        mover = json.load(f)["mover"]
    by_id = {c["id"]: c for c in cands}
    rules = {r["id"]: r for r in _jsonl(p_rules)}
    rc = np.load(p_cases)
    row = {int(i): j for j, i in enumerate(rc["ids"])}

    # -- steps 1-4 on every admitted-but-not-trusted rule ---------------------------------------------------
    records, new = [], []
    for rid, r in rules.items():
        if r["v2_admitted"] and not r["v2_trusted"]:
            rec, made = fit_rule(by_id[rid], rc["applies"][row[rid]], rc["claimed_ok"][row[rid]], train.actions,
                                 bits, names, mover, settings)
            records.append(rec)
            new += made
    # -- step 5: the replay ----------------------------------------------------------------------------------
    with ThreadPoolExecutor(WORKERS) as ex:
        graded = list(ex.map(lambda c: replay(c, p_train, train), new))
    res_of = {}
    for c, (res, t) in zip(new, graded):
        if res is None:
            c.update(stage="failed", reason=t, trust=None)
            continue
        res_of[c["id"]] = res
        c.update(stage="checked", trust=t, trusted=t["trusted"], noop_exact=noop_exact(t),
                 reason=t["reason"])
    for rec in records:
        kids = [c for c in new if c["parent"] == rec["id"]]
        rec["replay"] = {c["kind"]: {"trusted": c.get("trusted", False), "noop_exact": c.get("noop_exact", False),
                                     "reason": c["reason"]} for c in kids}

    # -- the book on the training level, before and after (CP1's order: cp1_import.py) ------------------------
    order = {g: i for i, g in enumerate(book_before["groups_called"])}
    in_book = lambda c: c.get("origin") != "two-level" and not c.get("duplicate")
    key = lambda c: (order.get(c["group"], len(order)), bool(c["lint"]), -c["trust"]["gain_cells"], c["order"])
    old = sorted((by_id[i] for i, r in rules.items() if r["v2_trusted"] and in_book(by_id[i])), key=key)
    rescued = [c for c in new if c["kind"] == "narrowed" and c.get("trusted")]
    offered = {"before": old, "after": sorted(old + [c for c in rescued if in_book(c)], key=key)}
    books_out = {}
    for when, cs in offered.items():
        comp = K.compose([member(c, p_train) for c in cs], p_train, select=True)
        if comp["stage"] != "checked":
            raise RuntimeError(f"{game}: the book {when} could not be composed: {comp.get('reason')}")
        books_out[when] = {"rules": [cs[i] for i in comp["book"]], "numbers": K.numbers(comp, train)}

    # -- T1 on the next level, before and after ------------------------------------------------------------------
    t1 = {}
    if os.path.exists(p_score):
        fit, score = LevelEvidence.load(p_fit), LevelEvidence.load(p_score)
        base = {"nothing": moves_right(score, nothing_hash(score)),
                "memory": moves_right(score, memory_hash([train, fit], score))}
        fresh = [c for c in books_out["after"]["rules"] if c["id"] in res_of]
        with ThreadPoolExecutor(WORKERS) as ex:
            for c, got in zip(fresh, ex.map(lambda c: carry(c, res_of[c["id"]], train, fit, p_fit), fresh)):
                c["t1"] = got
        for when in ("before", "after"):
            kept = [c for c in books_out[when]["rules"] if (c.get("t1") or {}).get("in_t1")]
            t1[when], _ = score_book([member(c, p_fit, c["t1"]["binding"]) for c in kept], p_score, score, base)
            t1[when]["members"] = [c["id"] for c in kept]

    # -- "before" must be what the import recorded ---------------------------------------------------------------
    matches = {"book": plain(books_out["before"]["numbers"]) == plain(book_before["numbers"])
               if "numbers" in book_before else None, "t1": None}
    if t1 and os.path.exists(p_carry):
        with open(p_carry) as f:
            recorded = json.load(f)["t1"]
        matches["t1"] = plain({x: v for x, v in t1["before"].items() if x != "members"}) == plain(recorded)
    if False in matches.values():
        log(f"[{game}] MISMATCH: 'before' differs from what the import recorded: {matches}")

    # -- files -------------------------------------------------------------------------------------------------
    os.makedirs(os.path.join(s3, game), exist_ok=True)
    p_guards, p_new = os.path.join(s3, game, "guards.jsonl"), os.path.join(s3, game, "guarded.jsonl")
    with open(p_guards, "w") as f:
        for rec in records:
            f.write(json.dumps(plain(rec)) + "\n")
    with open(p_new, "w") as f:
        for c in new:
            f.write(json.dumps(plain(c)) + "\n")
    before_ex = {r["id"]: r["trust"]["claimed_exactness"] for r in rules.values()}
    summary = {
        "game": game, "level": k, "transfer_level": k + 1 if t1 else None, "settings": settings,
        "looked_at": len(records), "guards": sum(r["guard"] is not None for r in records),
        "why_none": dict(collections.Counter(r["why"] for r in records if r["guard"] is None)),
        "guards_found": [{"id": r["id"], "group": r["group"], "guard": r["words"], "right": r["right"],
                          "wrong": r["wrong"], "replay": r["replay"]} for r in records if r["guard"] is not None],
        "rescued": [{"id": c["id"], "parent": c["parent"], "group": c["group"], "guard": c["words"],
                     "exactness_before": before_ex[c["parent"]], "exactness_after": c["trust"]["claimed_exactness"],
                     "moves_before": rules[c["parent"]]["trust"]["moves"], "moves_after": c["trust"]["moves"],
                     "in_book": any(c is b for b in books_out["after"]["rules"]),
                     "in_t1": bool((c.get("t1") or {}).get("in_t1"))} for c in rescued],
        "exceptions_exact": [c["id"] for c in new if c["kind"] == "exception" and c.get("noop_exact")],
        "book": {w: {"members": [c["id"] for c in b["rules"]], **b["numbers"]} for w, b in books_out.items()},
        "t1": t1, "before_matches_import": matches}
    p_sum = os.path.join(s3, f"{game}_L{k}.guards.json")
    with open(p_sum, "w") as f:
        json.dump(plain(summary), f, indent=1)
    run.write_header("s3_check", name, VERSION, settings, inputs,
                     {"guards": p_guards, "guarded": p_new, "summary": p_sum}, code=__file__)
    b0, b1 = summary["book"]["before"], summary["book"]["after"]
    log(f"[{game}] {summary['looked_at']} near-miss rules, {summary['guards']} guards, "
        f"{len(rescued)} rescued (trusted), {len(summary['exceptions_exact'])} exact exceptions; book cell coverage "
        f"{b0['cell_coverage']:.1%} -> {b1['cell_coverage']:.1%}, case coverage {b0['case_coverage']:.1%} -> "
        f"{b1['case_coverage']:.1%}" + (f"; T1 {t1['before']['per_move']['rate']['book']:.1%} -> "
                                         f"{t1['after']['per_move']['rate']['book']:.1%}" if t1 else ""))
    return summary
