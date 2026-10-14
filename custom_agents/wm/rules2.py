"""rules2.py - what the LLM is shown and asked under the Rulebook v2 contract
(docs/plans/rulebook-v2.md sections 4.1, 4.4-4.6). New prompts only: v1's text in
wm/rules.py is not touched (its answer cache is keyed on it), and its pictures
and move descriptions are reused from there as they are.

Four prompts, all for ONE action group:
  first round   v1's round-1 content (the level's first board, up to 6 recorded
                moves) with the v2 contract: PARAMS, bind(), claimed cells.
  refinement    for the bandit (section 4.6): 3 recorded moves, one rule with
                this level's binding and its v2 scores, the lint's warnings, and
                up to 6 moves it fails on - claimed cells it got wrong first,
                then moves of the group it did not cover.
  residual      for a partly explained group (section 4.5): the trusted rules'
                one-line summaries, and moves where changed cells are still
                unclaimed, with only THOSE cells boxed. Asks for a rule for them.
  two-level     after the next level's fit window (section 4.4): up to 3 moves
                from each level, the rule, how its re-binding went, its errors.
                Asks for one mechanism for both levels.

A prompt may hold at most 10 pictures (the server's limit) and 12,288 tokens (so
a 20,480-token answer fits beside it): fit() drops examples from the end of a
prompt's optional part until both hold, by a deliberately high token estimate.
"""
import collections

import numpy as np

from . import contract as C
from . import lint as L
from . import rules as R
from .bind import family_of
from .evidence import PALETTE_NAMES

N_SHOWN, N_FAILING = 3, 6          # recorded moves and failing moves in a refinement prompt
N_PER_LEVEL = 3                    # moves per level in a two-level prompt
N_RESIDUAL = 5                     # moves in a residual prompt
MAX_IMAGES, MAX_PROMPT_TOKENS = 10, 11500

RULE_DOC = """You write ONE rule about how this game works, as Python. `np` (numpy) is \
available; nothing can be imported.

    RULE = "one line saying what happens"
    PARAMS = {"agent": Colour(), "step": Int(1, 8)}   # every level-specific fact, named by its role
    def bind(board0, api):
        # optional: your guess at PARAMS for a level, from its first board
        return {"agent": ..., "step": ...}
    def applies(board, act, api, p):
        # True if your rule knows what this action does on this board
        ...
    def predict(board, act, api, p):
        # the board right AFTER the action, and the cells you vouch for
        out = board.copy()
        ...
        return out, claimed

`board` is the board BEFORE the action: a 64x64 numpy array of colour indices 0-15.
It is read-only, so work on board.copy().
`act.action` is 1-5 for a button (ACTION1-ACTION5) or 6 for a click; `act.click`
is (row, col) for a click and None otherwise.
`p` holds this level's value for every name in PARAMS: p["agent"], p["step"].
`claimed` is a 64x64 bool array: True on the cells your rule vouches for. Every
other cell is left unknown.
PARAMS must be a literal dict as above. Each value is one of:
  Colour()         a colour index
  Int(lo, hi)      a whole number from lo to hi
  Offset(max_abs)  a (drow, dcol) pair, each from -max_abs to max_abs (max_abs up to 8)
  Region()         one region of api.layout; p[name] is the region itself
  Choice([...])    one of the listed values
You do not pick the values. On each level they are found by trying every
combination against the recorded moves (so keep PARAMS small: at most 4,096
combinations), starting with bind()'s guess.
`api` gives you:
  api.first       the board at the START of this level (64x64)
  api.background  the most common colour of api.first
  api.colours     the colours on api.first
  api.level       the level's number (1 for the first)
  api.ticker      64x64 bool: timer/counter cells that change by themselves. They
                  are ignored when your rule is checked.
  api.layout      regions on api.first: connected same-colour areas (background and
                  ticker excluded), largest first. Each region r has r.colour,
                  r.size, r.y0, r.x0, r.y1, r.x1 (inclusive bounding box), r.mask
                  (64x64 bool, True on the region) and r.cells (n x 2 (row, col)).

Levels of this game share their mechanics but differ in colours, positions and
sizes. Write the mechanism once, put anything that could differ into PARAMS, and
claim only the cells you can explain.

How rules are checked: on every recorded move where applies() is True, every cell
you claim must match the real next board EXACTLY. A rule is kept if it applies to
at least 20 recorded moves, its claimed cells are all right on at least 95% of
them, and it predicts at least 20 changed cells right in total; it is trusted only
if its claimed cells are right on ALL of them. So:
- make applies() narrow enough that the rule is right whenever it says it applies;
- claim the cells your rule changes, and nothing you cannot explain: a wrong claimed
  cell fails the move, and a rule that claims almost nothing is worth nothing;
- never write a colour number or a board position in the code: it is reported as a
  fault, because it will not hold on the next level. Use PARAMS, api.background, or
  find things on the board;
- keep it fast: whole-array numpy, no Python loops over all 4,096 cells.

Example, from a DIFFERENT game - it shows the style, not the answer here:

    RULE = "ACTION1 moves the agent block up by its step"
    PARAMS = {"agent": Colour(), "step": Int(1, 4)}
    def bind(board0, api):
        blocks = sorted(api.layout, key=lambda r: r.size)
        return {"agent": blocks[0].colour, "step": 1} if blocks else {}
    def applies(board, act, api, p):
        return act.action == 1 and (board == p["agent"]).any()
    def predict(board, act, api, p):
        out = board.copy()
        g = board == p["agent"]
        moved = np.roll(g, -p["step"], axis=0)
        out[g] = api.background
        out[moved] = p["agent"]
        return out, g | moved"""

SYSTEM = ("You work out the exact mechanics of an unknown grid game from recorded moves, "
          "and write each one down as a small Python rule.\n\n" + RULE_DOC +
          f"\n\nColour numbers are drawn as: {R.LEGEND}.")
DEFINES = "RULE, PARAMS, applies and predict (and bind, if the first board is enough to guess the values)"
TASK = ("Write ONE rule for this action group: what exactly does this action do to the board? "
        "Your rule is checked on ALL recorded moves of this level, not only the ones shown, so "
        "applies() must return False for moves it does not describe. It is then used on the game's "
        "other levels with PARAMS fitted again, so keep every colour and position out of the code.")
ANSWER_FORMAT = ("Think it through, but keep it brief: an answer that runs out of room before the "
                 f"code is wasted. Then give exactly one ```python code block that defines {DEFINES}.")
REPAIR = ("Your rule could not be used: {reason}\nFix it and answer with exactly one ```python "
          f"code block that defines {DEFINES}.")
FINISH = ("You ran out of room before writing the code. Write it now, with no further analysis: "
          f"exactly one ```python code block that defines {DEFINES}.")
IMPROVE = ("Write ONE improved rule for this group. Every cell it claims must be right on every move it "
           "covers, including the ones shown above: keep what the rule gets right and fix what it gets "
           "wrong. If a cell cannot be predicted from the board, do not claim it; if a whole move cannot "
           "be, make applies() return False for it.")
DELITERAL = ("This rule is right on every move of this level, but it has level facts written into its "
             "code, so it cannot be used on the next level. Write it again so that it does exactly the "
             "same thing with every one of those facts in PARAMS.")
AGAIN = ("None of the rules written for this group so far could be used. Write ONE rule for it that "
         "runs and follows the contract.")
REST = ("The rules above are trusted, but they leave changed cells unexplained. Write ONE new rule for "
        "the boxed changes alone: applies() true on moves like these, and claim only the cells this new "
        "rule explains. Do not repeat what the rules above already claim.")
BOTH = ("Write ONE rule that describes the same mechanism on BOTH levels. Whatever differs between the "
        "two levels (colours, positions, sizes, distances) must be a PARAMS slot, so that fitting PARAMS "
        "to each level makes the rule right on both.")


# --------------------------------------------------------------------------------
# Keeping a prompt inside the server's limits
# --------------------------------------------------------------------------------
def estimate_tokens(parts):
    """A deliberately high estimate of a prompt's size: 1 token per 3 characters
    of text, and a picture's tiles (wm/rules.py: a 384x384 board is 144)."""
    n = len(SYSTEM) // 3 + 200
    for p in parts:
        if p["type"] == "text":
            n += len(p["text"]) // 3 + 4
        else:
            n += p.get("_tokens", 450)
    return n


def picture(img):
    """wm.rules.image_part, remembering how many tokens the picture costs."""
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    else:
        w, h = img.size
    part = R.image_part(img)
    part["_tokens"] = int(np.ceil(w / 32) * np.ceil(h / 32)) + 8
    return part


def fit(fixed, optional, tail):
    """fixed + as many of the `optional` examples (each a list of parts) as fit
    + tail, inside MAX_IMAGES and MAX_PROMPT_TOKENS. Returns (parts, examples kept)."""
    count = lambda parts: sum(p["type"] != "text" for p in parts)
    kept = list(optional)
    while kept:
        parts = fixed + [p for ex in kept for p in ex] + tail
        if count(parts) <= MAX_IMAGES and estimate_tokens(parts) <= MAX_PROMPT_TOKENS:
            break
        kept.pop()
    return fixed + [p for ex in kept for p in ex] + tail, len(kept)


def conversation(parts, instruction=TASK):
    """The request: v2's system prompt, the prompt's parts, the instruction."""
    clean = [{k: v for k, v in p.items() if not k.startswith("_")} for p in parts]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": clean + [R.text_part(instruction + "\n\n" + ANSWER_FORMAT)]}]


# --------------------------------------------------------------------------------
# Pieces shared by the prompts
# --------------------------------------------------------------------------------
def header(ev, group, rng, n_moves):
    """The level's first board, the group's statistics and `n_moves` recorded
    moves: v1's round-1 content (wm.rules.group_prompt), cut to n_moves."""
    parts = R.group_prompt(ev, group, rng)
    head, moves = parts[:3], parts[3:]
    stats = head[2] if n_moves else R.text_part(head[2]["text"].split("\n\nRecorded moves")[0])
    out = [head[0], picture(R.render_board(ev.first)), stats]
    for j in range(min(n_moves, len(moves) // 2)):
        out += [moves[2 * j], dict(moves[2 * j + 1], _tokens=305)]
    return out


def verdict(arm):
    """One rule's scores on its own level, in the v2 contract's terms."""
    t, g = arm["trust"], arm["grade"]
    how = ("TRUSTED: its claimed cells are right on every move." if t["trusted"] else
           "kept, but not right on every move." if t["admitted"] else f"not kept: {t['reason']}.")
    fitted = (f"With PARAMS fitted to this level ({arm['binding_text']})" if arm.get("exact_binding") else
              f"No values of PARAMS make it right on every move. With the best found ({arm['binding_text']})")
    if not arm.get("has_params"):
        fitted = "It"
    return (f"RULE: {arm['rule']}\n{fitted} it applies to {t['moves']} recorded moves "
            f"({g['in_group_moves']} of this group's {g['group_moves']}); its claimed cells are all right on "
            f"{t['claimed_exactness']:.0%} of them; it predicts {t['cells_right']} changed cells right and "
            f"claims {t['cells_false']} changes that did not happen - {how}")


def wrong_case(ev, k, pred, claim):
    """A move where a claimed cell was wrong: text + BEFORE | PREDICTION | AFTER."""
    off = ~np.asarray(claim, bool) | ev.mask                    # only claimed cells are held against the rule
    areas = R._areas(pred, ev.after[k], off, lambda p, q, n: f"predicted {p}, really {q} x{n}", 12)
    return [R.text_part(f"The rule was WRONG on this move: {R.move_text(ev, k)}\nClaimed cells that do not match "
                        "the real board:\n  " + "\n  ".join(areas) + "\nPicture: BEFORE | THE RULE'S PREDICTION | "
                        "REAL AFTER (cyan boxes: claimed cells that are wrong)."),
            dict(R.image_part(R.render_compare(ev.before[k], pred, ev.after[k], off)), _tokens=455)]


def missed_case(ev, k):
    return [R.text_part("The rule did not apply to this move of the group, where the board DID "
                        f"change: {R.move_text(ev, k)}"),
            dict(R.image_part(R.render_move(ev.before[k], ev.after[k], ev.mask)), _tokens=305)]


def rule_block(arm):
    return f"```python\n{arm['code']}\n```"


# --------------------------------------------------------------------------------
# The prompts
# --------------------------------------------------------------------------------
def first_round(ev, group, rng):
    """Round 1: (parts, instruction)."""
    return header(ev, group, rng, R.N_EXAMPLES), TASK


def refinement(ev, group, arm, rng_examples, rng_cases):
    """One bandit step on one rule: (parts, instruction, failing moves shown)."""
    fixed = header(ev, group, rng_examples, N_SHOWN)
    text = ("A rule written for this group, and how it did on ALL recorded moves of this level:\n\n"
            + verdict(arm) + "\n" + rule_block(arm))
    warnings = L.feedback(arm.get("lint") or [])
    if warnings:
        text += "\n\n" + warnings
    fixed.append(R.text_part(text))
    cases = []
    wrong = arm["res"]["wrong"]
    for j in rng_cases.permutation(len(wrong))[:N_FAILING]:
        cases.append(wrong_case(ev, *wrong[j]))
    missed = arm["grade"].get("missed") or []
    for j in rng_cases.permutation(len(missed))[:max(N_FAILING - len(cases), 0)]:
        cases.append(missed_case(ev, int(missed[j])))
    lint_only = arm["trust"]["trusted"] and bool(warnings)
    parts, shown = fit(fixed, cases, [])
    return parts, (DELITERAL if lint_only else IMPROVE), shown


def retry(ev, group, failures, rng):
    """A refinement call for a group with no rule to refine: round 1 again, with
    why the earlier candidates could not be used. (parts, instruction)."""
    why = collections.Counter(f[:120] for f in failures)
    lines = ["Rules written for this group so far could not be used:"] + [f"  {n} x {w}" for w, n in why.most_common(5)]
    return header(ev, group, rng, R.N_EXAMPLES - 1) + [R.text_part("\n".join(lines))], AGAIN


def residual(ev, group, trusted, open_cells, rng):
    """For a partly explained group: (parts, instruction, moves shown).
    `trusted`: the group's trusted rules; `open_cells`: {case: 64x64 bool of
    changed cells that no rule in the book claims}."""
    fixed = header(ev, group, rng, 0)
    fixed.append(R.text_part("Trusted rules for this group so far (each is right on every move it covers):\n"
                             + "\n".join(f"- {a['rule']}" for a in trusted)
                             + "\n\nOn the moves below, some changed cells are claimed by none of them. "
                             "Each picture shows the board BEFORE the move (left) and AFTER it (right), with a "
                             "cyan box around the UNEXPLAINED changes only."))
    keys = [k for k in open_cells if ev.group[k] == group and not ev.frozen[k]]
    by_sig = collections.defaultdict(list)                      # different kinds of leftover first
    for k in keys:
        cells = open_cells[k]
        flows = collections.Counter(zip(ev.before[k][cells].tolist(), ev.after[k][cells].tolist()))
        by_sig[(int(cells.sum()), tuple(sorted(flows.items())))].append(k)
    sigs = sorted(by_sig, key=lambda s: (-sum(int(ev.count[k]) for k in by_sig[s]), s))
    for s in sigs:                                              # random order within a kind
        by_sig[s] = [by_sig[s][j] for j in rng.permutation(len(by_sig[s]))]
    pick, depth, want = [], 0, min(N_RESIDUAL, len(keys))
    while len(pick) < want:                                     # round-robin over kinds
        for s in sigs:
            if depth < len(by_sig[s]) and len(pick) < want:
                pick.append(int(by_sig[s][depth]))
        depth += 1
    cases = []
    for j, k in enumerate(pick, 1):
        cells = open_cells[k]
        areas = R._areas(ev.before[k], ev.after[k], ~cells, lambda p, q, n: f"{p}->{q} x{n}", 12)
        cases.append([R.text_part(f"Move {j}: {R.move_text(ev, k)}\nUnexplained changes:\n  " + "\n  ".join(areas)),
                      dict(R.image_part(R._boxes(*R._panels((ev.before[k], ev.after[k]), R.CELL), cells, (0, 1),
                                                 R.CELL, R.AREAS_SHOWN)), _tokens=305)])
    parts, shown = fit(fixed, cases, [])
    return parts, REST, shown


def family_examples(ev, family, rng, n):
    """Up to n changing cases of a family on one level, covering distinct effects."""
    keys = [k for k in range(len(ev.actions)) if family_of(ev.group[k]) == family and ev.changed[k] and not ev.frozen[k]]
    by_sig = collections.defaultdict(list)
    for k in keys:
        by_sig[R.effect_signature(ev, k)].append(k)
    sigs = sorted(by_sig, key=lambda s: (-sum(int(ev.count[k]) for k in by_sig[s]), s))
    pick = [int(by_sig[s][int(rng.integers(0, len(by_sig[s])))]) for s in sigs[:n]]
    rest = [k for k in keys if k not in pick]
    pick += [int(rest[j]) for j in rng.permutation(len(rest))[:n - len(pick)]]
    return pick


def two_level(home, new, group, arm, rebind, rng):
    """After the next level's fit window: (parts, instruction).
    `home`: the level the rule was written on; `new`: the next level's FIT split;
    `rebind`: {flag, text, binding_text, wrong: [(case, prediction, claim)]} from
    fitting the rule's PARAMS to `new`."""
    family = family_of(group)
    kind = group if family != "click" else "clicks"
    fixed = [R.text_part(f"LEVEL {home.level + 1} - the level the rule below was written on. Its first board:"),
             picture(R.render_board(home.first)),
             R.text_part(f"{R.ticker_line(home.mask)}\n\nACTION GROUP on level {home.level + 1}: {group}.\n"
                         f"Recorded moves on level {home.level + 1} (BEFORE left, AFTER right; a cyan box marks "
                         "what changed):")]
    for j, k in enumerate(family_examples(home, family, rng, N_PER_LEVEL), 1):
        fixed += [R.text_part(f"Level {home.level + 1}, move {j}: {R.move_text(home, k)}"),
                  dict(R.image_part(R.render_move(home.before[k], home.after[k], home.mask)), _tokens=305)]
    fixed += [R.text_part(f"LEVEL {new.level + 1} - the next level of the same game. Its first board:"),
              picture(R.render_board(new.first)),
              R.text_part(f"{R.ticker_line(new.mask)}\n\nThe first recorded moves of the same kind ({kind}) on "
                          f"level {new.level + 1}:")]
    for j, k in enumerate(family_examples(new, family, rng, N_PER_LEVEL), 1):
        fixed += [R.text_part(f"Level {new.level + 1}, move {j}: {R.move_text(new, k)}"),
                  dict(R.image_part(R.render_move(new.before[k], new.after[k], new.mask)), _tokens=305)]
    text = (f"The rule, as written on level {home.level + 1}:\n\n" + verdict(arm) + "\n" + rule_block(arm)
            + f"\n\nOn level {new.level + 1}: {rebind['text']}")
    warnings = L.feedback(arm.get("lint") or [])
    if warnings:
        text += "\n\n" + warnings
    fixed.append(R.text_part(text))
    cases = [wrong_case(new, *w) for w in (rebind.get("wrong") or [])[:2]]
    parts, _ = fit(fixed, cases, [])
    return parts, BOTH


def colour_name(c):
    return f"{c} ({PALETTE_NAMES[c]})" if 0 <= c < len(PALETTE_NAMES) else str(c)


def binding_text(binding, params, layout):
    """A binding as the prompts show it, with colour names."""
    if not binding:
        return "no parameters"
    parts = []
    for name, v in binding.items():
        if isinstance(params.get(name), C.Colour):
            parts.append(f"{name} = {colour_name(int(v))}")
        else:
            parts.append(C.describe({name: v}, params, layout))
    return ", ".join(parts)
