"""s2_write.py - S2 WRITE: the local LLM writes one small rule per mechanism, one action group at a time.

The problem it fixes: in CP1 one rule often tried to explain several mechanisms
at once (a move, a blocked move, a bar), its code was long raw numpy, and 44% of
its answers ran out of room before finishing. (Plan section 5.2.)

What it does, version 1 (the plan's cut line): Rulebook v2's procedure for each
action group of the training level (legacy/rulebook/tools/wm_tier0a_v2.py,
run_group), held equal to CP1 (4 first-round candidates, then up to 8 calls
chosen by v2's bandit on claimed-cell exactness, then residual prompts once the
group has a trusted rule; thinking on, 20,480-token answers, one code-only
retry, one repair; Qwen's sampling), with four changes:

  1. FACTS. Under every move shown, S1's description of it (s1_see facts text):
     the objects that changed, and what is next to the object the move is about.
  2. VOCABULARY. H1's card (h1_vocab.CARD) in the instructions: the tested grid
     operations the rule may call.
  3. ONE MECHANISM. The instruction asks for the commonest effect only, other
     effects (bars, blocked moves, side effects) being left to their own rules.
  4. GUARDS FIRST. Before a call is spent refining a near-miss (claimed cells
     right on >= 95% of its moves), S3's guard fitter (s3_guards.fit_rule, no
     LLM) tries it once. If the narrowed rule it writes is trusted, no call is
     spent; otherwise the call refines the rule as before.

Not built, by the plan's cut line: exception prompts (`exceptions = false`),
two-level prompts, and the optional KIND / PRIORITY fields (no priority
composition exists yet). Pictures, move text and the prompts' limits (at most
10 pictures and 11,500 estimated tokens, examples dropped first) are v2's,
copied from legacy/rulebook/wm/rules.py and rules2.py.

Every candidate's PARAMS are fitted to the level by search (bind.search) and
graded by v2's trust (book.trust), as in CP1. Answers are cached
(gb/core/llm.py), so a finished run can be re-scored without the GPU.

Reads  <run>/evidence/<game>_L<k>_train.npz, <run>/s1_see/<game>_L<k>_train.facts.{npz,jsonl}, .roles.json
Writes <run>/s2_write/<game>/candidates.jsonl   every candidate: code, binding, trust, origin, tokens
       <run>/s2_write/<game>/groups.json        the groups called, in order
       <run>/s2_write/<game>/llm.json           calls, tokens and seconds for the game
       <run>/s2_write/llm_cache.jsonl           every answer (all games)
"""
import ast
import base64
import collections
import io
import json
import os
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from gb import h1_vocab as H
from gb import s3_guards
from gb.artifacts import plain
from gb.core import bind as B
from gb.core import book as K
from gb.core import contract as C
from gb.core import lint as L
from gb.core.check import ADMIT, grade
from gb.core.evidence import GRID, PALETTE_NAMES, Act, LevelEvidence, connected_components

VERSION = 1                     # bump when what the output means changes (artifacts.py)
DEFAULTS = {"k_first": 4, "calls_per_group": 8, "max_groups": 12, "examples": 6, "facts": True,
            "exceptions": False, "thinking": True, "answer_tokens_offline": 20480,
            "code_only_retry": True, "repairs": 1}
GROUP_WORKERS = 4               # action groups worked on at the same time (CP1's)
BANDIT_C = 20.0                 # CP0's bandit
NEAR_MISS = 0.95                # claimed-exactness at which the guard fitter is tried before a call
BROKEN = ("static", "runtime", "timeout", "unbound")       # worth one repair attempt


def rng_for(*parts):
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


# =============================================================================================
# 1. Pictures and move text: copied unchanged from legacy/rulebook/wm/rules.py
# =============================================================================================
N_UNCHANGED = 2                    # of the moves shown, up to 2 that changed nothing
CELL = 6                           # px per cell: a board is 384x384 = 144 visual tokens
AREAS_SHOWN = 24                   # changed areas listed per move
MAX_TOKENS = {False: 4096, True: 20480}
SAMPLING = {False: dict(temperature=0.7, top_p=0.8, top_k=20),     # Qwen's non-thinking defaults
            True: dict(temperature=0.6, top_p=0.95, top_k=20)}     # Qwen's thinking defaults
PALETTE_HEX = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
               "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
               "#FF851B", "#921231", "#4FCC30", "#A356D6"]
PAL = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PALETTE_HEX], np.uint8)
GAP = 24                           # px gutter between panels
GUTTER = (58, 58, 90)              # dark slate: not a game colour
BOX = (0, 255, 255)                # cyan: not a game colour
LEGEND = ", ".join(f"{i} {n}" for i, n in enumerate(PALETTE_NAMES))


def render_board(frame, cell=CELL):
    rgb = PAL[np.asarray(frame, dtype=np.uint8)]
    return np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)


def _panels(frames, cell):
    side = GRID * cell
    img = Image.new("RGB", (len(frames) * side + (len(frames) - 1) * GAP, side), GUTTER)
    for k, f in enumerate(frames):
        img.paste(Image.fromarray(render_board(f, cell)), (k * (side + GAP), 0))
    return img, side


def _boxes(img, side, diff, panels, cell, limit):
    draw = ImageDraw.Draw(img)
    for c in sorted(connected_components(diff), key=len, reverse=True)[:limit]:
        (y0, x0), (y1, x1) = c.min(axis=0), c.max(axis=0)
        for k in panels:
            ox = k * (side + GAP)
            draw.rectangle([ox + max(int(x0) * cell - 2, 0), max(int(y0) * cell - 2, 0),
                            ox + min((int(x1) + 1) * cell + 1, side - 1),
                            min((int(y1) + 1) * cell + 1, side - 1)], outline=BOX, width=2)
    return img


def render_move(before, after, mask, cell=CELL):
    """BEFORE | AFTER, with a cyan box around each changed non-ticker area."""
    img, side = _panels((before, after), cell)
    return _boxes(img, side, (before != after) & ~mask, (0, 1), cell, AREAS_SHOWN)


def render_compare(before, pred, actual, mask, cell=CELL):
    """BEFORE | the rule's prediction | the real AFTER, boxing where the last two differ."""
    img, side = _panels((before, pred, actual), cell)
    return _boxes(img, side, (pred != actual) & ~mask, (1, 2), cell, 8)


def image_part(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def text_part(s):
    return {"type": "text", "text": s}


def _shape_key(cells):
    y0, x0 = cells[:, 0].min(), cells[:, 1].min()
    return frozenset((int(y - y0), int(x - x0)) for y, x in cells)


def detect_moves(before, after, changed):
    """Rigid translations of same-coloured shapes, per colour: [(colour, size, drow, dcol)]."""
    out = []
    for c in np.union1d(before[changed], after[changed]).tolist():
        left, arrived = changed & (before == c), changed & (after == c)
        n = int(left.sum())
        if n == 0 or n != int(arrived.sum()):
            continue
        lc, ac = np.argwhere(left), np.argwhere(arrived)
        if _shape_key(lc) == _shape_key(ac):
            out.append((c, n, int(ac[:, 0].min() - lc[:, 0].min()), int(ac[:, 1].min() - lc[:, 1].min())))
    return out


def _areas(a, b, mask, label, limit):
    """Connected areas where boards a and b differ, with their colour flows."""
    comps = sorted(connected_components((a != b) & ~mask), key=len, reverse=True)
    out = []
    for c in comps[:limit]:
        ys, xs = c[:, 0], c[:, 1]
        flow = collections.Counter(zip(a[ys, xs].tolist(), b[ys, xs].tolist()))
        out.append(f"rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()} ({len(c)} cells): "
                   + ", ".join(label(p, q, n) for (p, q), n in flow.most_common(4)))
    if len(comps) > limit:
        out.append(f"... and {len(comps) - limit} more areas ({sum(len(c) for c in comps[limit:])} cells)")
    return out


def move_text(ev, k):
    """One recorded move, exactly: the action, what was clicked, and every change."""
    b, a = ev.before[k], ev.after[k]
    act = Act(ev.actions[k])
    if act.click is None:
        head = f"ACTION{act.action}"
    else:
        y, x = act.click
        c = int(b[y, x])
        reg = next(r for r in connected_components(b == c) if ((r[:, 0] == y) & (r[:, 1] == x)).any())
        head = (f"click at (row {y}, col {x}) on colour {c} ({PALETTE_NAMES[c]}); the clicked "
                f"same-colour area is {len(reg)} cells, rows {reg[:, 0].min()}-{reg[:, 0].max()}, "
                f"cols {reg[:, 1].min()}-{reg[:, 1].max()}")
    changed = (b != a) & ev.live
    moved = [f"a {n}-cell colour-{c} shape moved by ({dy:+d} row, {dx:+d} col)"
             for c, n, dy, dx in detect_moves(b, a, changed)]
    changes = _areas(b, a, ev.mask, lambda p, q, n: f"{p}->{q} x{n}", AREAS_SHOWN) or \
        ["nothing changed (outside the ticker)"]
    return head + "\n  " + "\n  ".join(moved + changes)


def effect_signature(ev, k):
    """What kind of change a move made, position-free."""
    b, a = ev.before[k], ev.after[k]
    ch = (b != a) & ev.live
    flows = collections.Counter(zip(b[ch].tolist(), a[ch].tolist()))
    return (int(ch.sum()), tuple(sorted(flows.items())))


def ticker_line(mask):
    if not mask.any():
        return "This game has no ticker cells."
    ty, tx = np.nonzero(mask)
    return (f"Ticker cells (a timer/counter that changes by itself; ignored when rules are "
            f"checked): {int(mask.sum())} cells in rows {ty.min()}-{ty.max()}, cols "
            f"{tx.min()}-{tx.max()}.")


def pick_examples(ev, group, rng, n_examples):
    """Keys to show for a group: changing moves covering distinct effects (the
    most recorded effect first), then up to N_UNCHANGED that changed nothing."""
    keys = [k for k in ev.keys_of(group) if not ev.frozen[k]]
    changing = [k for k in keys if ev.changed[k]]
    unchanged = [k for k in keys if not ev.changed[k]]
    by_sig = collections.defaultdict(list)
    for k in changing:
        by_sig[effect_signature(ev, k)].append(k)
    sigs = sorted(by_sig, key=lambda s: (-sum(int(ev.count[k]) for k in by_sig[s]), s))
    n_unch = min(N_UNCHANGED, len(unchanged))
    for s in sigs:                                            # random order within an effect
        by_sig[s] = [by_sig[s][j] for j in rng.permutation(len(by_sig[s]))]
    want, pick, depth = min(n_examples - n_unch, len(changing)), [], 0
    while len(pick) < want:                                   # round-robin over effects
        for s in sigs:
            if depth < len(by_sig[s]) and len(pick) < want:
                pick.append(int(by_sig[s][depth]))
        depth += 1
    pick += [int(unchanged[j]) for j in rng.permutation(len(unchanged))[:n_unch]]
    return [pick[j] for j in rng.permutation(len(pick))]


def extract_code(text):
    """The last ```python block of the final answer (after any thinking)."""
    answer = (text or "").split("</think>")[-1]
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", answer, re.S)
    return blocks[-1].strip() if blocks else None


def tail(text, limit=24000):
    """What a retry is shown of the first answer (its last `limit` characters)."""
    think, _, answer = (text or "").rpartition("</think>")
    answer = answer.strip() or think.strip()
    return answer if len(answer) <= limit else "[...earlier reasoning omitted...]\n" + answer[-limit:]


# =============================================================================================
# 2. What the LLM is told: v2's contract (rules2.RULE_DOC) with the vocabulary and the facts
# =============================================================================================
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

ONE MECHANISM PER RULE. A game action often does several things: a move, a move
that is blocked, a bar that shrinks, a side effect. Write the rule for ONE of them
(the commonest), make applies() False where it does not happen, and leave the
others to their own rules. A short rule that is always right is worth more than a
long one that tries to cover everything.

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
- keep it fast: whole-array numpy, no Python loops over all 4,096 cells."""

FACTS_DOC = """Under each recorded move, "Seen as objects" is what a program found on
the screen: the object the move is about (the mover for a button, the clicked
object for a click) with what is next to it in each direction, and every object
that moved, changed colour, grew, shrank, vanished or appeared. Objects are
4-connected areas of one colour. Use these facts; they are exact."""


def system_prompt(settings, vocab=True):
    parts = ["You work out the exact mechanics of an unknown grid game from recorded moves, "
             "and write each one down as a small Python rule.", RULE_DOC]
    if vocab:
        parts.append(H.CARD)
    if settings["facts"]:
        parts.append(FACTS_DOC)
    parts.append(f"Colour numbers are drawn as: {LEGEND}.")
    return "\n\n".join(parts)


DEFINES = "RULE, PARAMS, applies and predict (and bind, if the first board is enough to guess the values)"
TASK = ("Write ONE rule for ONE mechanism of this action group: the commonest thing this action does to "
        "the board. Your rule is checked on ALL recorded moves of this level, not only the ones shown, so "
        "applies() must return False for moves it does not describe. It is then used on the game's other "
        "levels with PARAMS fitted again, so keep every colour and position out of the code.")
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


# =============================================================================================
# 3. The prompts (rules2.py's, with the facts under each move)
# =============================================================================================
N_SHOWN, N_FAILING, N_RESIDUAL = 3, 6, 5
MAX_IMAGES, MAX_PROMPT_TOKENS = 10, 11500


class Prompter:
    """Builds the prompts for one level: its evidence, S1's text per case, the settings."""

    def __init__(self, ev, facts_text, settings, vocab=True):
        self.ev, self.facts, self.settings = ev, facts_text, settings
        self.system = system_prompt(settings, vocab)

    def seen(self, k):
        if not self.settings["facts"] or k not in self.facts:
            return ""
        return "\n  Seen as objects:\n" + "\n".join("  " + line for line in self.facts[k])

    def estimate_tokens(self, parts):
        """A deliberately high estimate: 1 token per 3 characters, and each picture's tiles."""
        n = len(self.system) // 3 + 200
        for p in parts:
            n += len(p["text"]) // 3 + 4 if p["type"] == "text" else p.get("_tokens", 450)
        return n

    def fit(self, fixed, optional, tail_parts):
        count = lambda parts: sum(p["type"] != "text" for p in parts)
        kept = list(optional)
        while kept:
            parts = fixed + [p for ex in kept for p in ex] + tail_parts
            if count(parts) <= MAX_IMAGES and self.estimate_tokens(parts) <= MAX_PROMPT_TOKENS:
                break
            kept.pop()
        return fixed + [p for ex in kept for p in ex] + tail_parts, len(kept)

    def conversation(self, parts, instruction=TASK):
        clean = [{k: v for k, v in p.items() if not k.startswith("_")} for p in parts]
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": clean + [text_part(instruction + "\n\n" + ANSWER_FORMAT)]}]

    def header(self, group, rng, n_moves):
        """The level's first board, the group's statistics and up to n_moves
        recorded moves (each with S1's facts); moves are dropped first by fit()."""
        ev = self.ev
        keys = ev.keys_of(group)
        moves = int(ev.count[keys].sum())
        ch_moves = int(ev.count[keys][ev.changed[keys]].sum())
        sizes = [int(((ev.before[k] != ev.after[k]) & ev.live).sum()) for k in keys if ev.changed[k]]
        stats = (f"The recordings have {moves} moves in this group on {len(keys)} different boards; the "
                 f"board changed (outside the ticker) on {ch_moves / max(moves, 1):.0%} of the moves"
                 + (f", median {int(np.median(sizes))} cells changed." if sizes else "."))
        intro = (f"{ticker_line(ev.mask)}\n\nACTION GROUP: {group}.\n{stats}"
                 + ("\n\nRecorded moves of this group. Each picture shows the board BEFORE the move (left) "
                    "and AFTER it (right); a cyan box marks what changed." if n_moves else ""))
        fixed = [text_part("The board at the start of this level:"), self.picture(render_board(ev.first)),
                 text_part(intro)]
        examples = [[text_part(f"Move {j}: {move_text(ev, k)}{self.seen(k)}"),
                     dict(image_part(render_move(ev.before[k], ev.after[k], ev.mask)), _tokens=305)]
                    for j, k in enumerate(pick_examples(ev, group, rng, self.settings["examples"])[:n_moves], 1)]
        return fixed, examples

    @staticmethod
    def picture(img):
        h, w = img.shape[:2]
        part = image_part(img)
        part["_tokens"] = int(np.ceil(w / 32) * np.ceil(h / 32)) + 8
        return part

    def first_round(self, group, rng):
        fixed, examples = self.header(group, rng, self.settings["examples"])
        return self.fit(fixed, examples, [])[0], TASK

    def wrong_case(self, k, pred, claim):
        ev = self.ev
        off = ~np.asarray(claim, bool) | ev.mask
        areas = _areas(pred, ev.after[k], off, lambda p, q, n: f"predicted {p}, really {q} x{n}", 12)
        return [text_part(f"The rule was WRONG on this move: {move_text(ev, k)}{self.seen(k)}\nClaimed cells that "
                          "do not match the real board:\n  " + "\n  ".join(areas) + "\nPicture: BEFORE | THE RULE'S "
                          "PREDICTION | REAL AFTER (cyan boxes: claimed cells that are wrong)."),
                dict(image_part(render_compare(ev.before[k], pred, ev.after[k], off)), _tokens=455)]

    def missed_case(self, k):
        ev = self.ev
        return [text_part("The rule did not apply to this move of the group, where the board DID "
                          f"change: {move_text(ev, k)}{self.seen(k)}"),
                dict(image_part(render_move(ev.before[k], ev.after[k], ev.mask)), _tokens=305)]

    def refinement(self, group, arm, rng_examples, rng_cases):
        fixed, examples = self.header(group, rng_examples, N_SHOWN)
        fixed = fixed + [p for ex in examples for p in ex]
        text = ("A rule written for this group, and how it did on ALL recorded moves of this level:\n\n"
                + verdict(arm) + "\n" + f"```python\n{arm['code']}\n```")
        warnings = L.feedback(arm.get("lint") or [])
        if warnings:
            text += "\n\n" + warnings
        fixed.append(text_part(text))
        cases = []
        wrong = arm["res"]["wrong"]
        for j in rng_cases.permutation(len(wrong))[:N_FAILING]:
            cases.append(self.wrong_case(*wrong[j]))
        missed = arm["grade"].get("missed") or []
        for j in rng_cases.permutation(len(missed))[:max(N_FAILING - len(cases), 0)]:
            cases.append(self.missed_case(int(missed[j])))
        parts, shown = self.fit(fixed, cases, [])
        return parts, (DELITERAL if arm["trust"]["trusted"] and warnings else IMPROVE), shown

    def retry(self, group, failures, rng):
        why = collections.Counter(f[:120] for f in failures)
        lines = ["Rules written for this group so far could not be used:"] + [f"  {n} x {w}" for w, n in
                                                                               why.most_common(5)]
        fixed, examples = self.header(group, rng, self.settings["examples"] - 1)
        return self.fit(fixed, examples, [text_part("\n".join(lines))])[0], AGAIN

    def residual(self, group, trusted, open_cells, rng):
        ev = self.ev
        fixed, _ = self.header(group, rng, 0)
        fixed.append(text_part("Trusted rules for this group so far (each is right on every move it covers):\n"
                               + "\n".join(f"- {a['rule']}" for a in trusted)
                               + "\n\nOn the moves below, some changed cells are claimed by none of them. "
                               "Each picture shows the board BEFORE the move (left) and AFTER it (right), with a "
                               "cyan box around the UNEXPLAINED changes only."))
        keys = [k for k in open_cells if ev.group[k] == group and not ev.frozen[k]]
        by_sig = collections.defaultdict(list)
        for k in keys:
            cells = open_cells[k]
            flows = collections.Counter(zip(ev.before[k][cells].tolist(), ev.after[k][cells].tolist()))
            by_sig[(int(cells.sum()), tuple(sorted(flows.items())))].append(k)
        sigs = sorted(by_sig, key=lambda s: (-sum(int(ev.count[k]) for k in by_sig[s]), s))
        for s in sigs:
            by_sig[s] = [by_sig[s][j] for j in rng.permutation(len(by_sig[s]))]
        pick, depth, want = [], 0, min(N_RESIDUAL, len(keys))
        while len(pick) < want:
            for s in sigs:
                if depth < len(by_sig[s]) and len(pick) < want:
                    pick.append(int(by_sig[s][depth]))
            depth += 1
        cases = []
        for j, k in enumerate(pick, 1):
            cells = open_cells[k]
            areas = _areas(ev.before[k], ev.after[k], ~cells, lambda p, q, n: f"{p}->{q} x{n}", 12)
            cases.append([text_part(f"Move {j}: {move_text(ev, k)}{self.seen(k)}\nUnexplained changes:\n  "
                                    + "\n  ".join(areas)),
                          dict(image_part(_boxes(*_panels((ev.before[k], ev.after[k]), CELL), cells, (0, 1), CELL,
                                                 AREAS_SHOWN)), _tokens=305)])
        parts, shown = self.fit(fixed, cases, [])
        return parts, REST, shown


def colour_name(c):
    return f"{c} ({PALETTE_NAMES[c]})" if 0 <= c < len(PALETTE_NAMES) else str(c)


def binding_text(binding, params, layout):
    if not binding:
        return "no parameters"
    parts = []
    for name, v in binding.items():
        if isinstance(params.get(name), C.Colour):
            parts.append(f"{name} = {colour_name(int(v))}")
        else:
            parts.append(C.describe({name: v}, params, layout))
    return ", ".join(parts)


def verdict(arm):
    """One rule's scores on its own level, in the contract's terms (rules2.verdict)."""
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


# =============================================================================================
# 4. Grading a candidate, and asking for candidates (CP1's evaluate and ask)
# =============================================================================================
def primitives_used(code):
    """The H1 names a rule's code calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return sorted(names & set(H.SANDBOX))


def graded_final(found, best, ev):
    """The binding a rule is graded and shown with (CP1's)."""
    if best is not None:
        return best
    finals = found.get("finals") or []
    applying = [f for f in finals if f["res"]["applies"].any()]
    if applying:
        return max(applying, key=lambda f: (K.trust([(f["res"], ev)])["claimed_exactness"],
                                            int(f["res"]["cells_right"].sum()), -f["order"]))
    return finals[0] if finals else None


def finish_arm(arm, code, ev, res, binding, exact, flag, group, found=None):
    _, params = C.read_params(code)
    arm.update(stage="checked", rule=res.get("rule") or (found or {}).get("rule", ""), has_params=bool(params),
               binding=binding, exact_binding=exact, flag=flag, binding_text=binding_text(binding, params,
                                                                                          ev.api.layout),
               res=res, grade=grade(res, ev, group), trust=K.trust([(res, ev)]),
               primitives=primitives_used(code))
    if found is not None:
        arm["search"] = {k: found[k] for k in ("bindings", "untried", "halving", "exact", "exact_applying",
                                              "guess", "guess_note", "unused")}
    arm["reason"] = arm["trust"]["reason"]
    return arm


def evaluate(code, group, ev, ev_path):
    """Fit a rule's PARAMS to the level by search and grade it there (CP1's evaluate)."""
    arm = {"code": code, "group": group, "lint": L.lint(code)}
    found = B.search(code, ev_path, family=B.family_of(group))
    arm.update(stage=found["stage"], reason=found.get("reason", "ok"))
    if found["stage"] != "checked":
        return arm
    best, flag = B.choose(found)
    final = graded_final(found, best, ev)
    if final is None:
        arm.update(stage="unbound")
        return arm
    return finish_arm(arm, code, ev, final["res"], final["binding"], best is not None, flag, group, found)


def evaluate_bound(code, group, ev, ev_path, binding):
    """Grade a rule with a given binding (a guard-narrowed rule: its parent's plus the guard's slots)."""
    arm = {"code": code, "group": group, "lint": L.lint(code)}
    res = B.run_rule(code, ev_path, binding=binding)         # the child reports the rule's text too
    if res.get("stage") != "checked":
        return dict(arm, stage=res.get("stage", "runtime"), reason=res.get("reason"))
    return finish_arm(arm, code, ev, res, binding, bool(res["claimed_ok"][res["applies"]].all()), None, group)


class Writer:
    """S2 on one level: the LLM client, the prompter, the evidence, S1's facts for the guard fitter."""

    def __init__(self, game, ev, ev_path, client, settings, facts_text, bits, names, mover, vocab=True,
                 log=print):
        self.game, self.ev, self.ev_path, self.client, self.settings, self.log = game, ev, ev_path, client, settings, log
        self.prompts = Prompter(ev, facts_text, settings, vocab)
        self.bits, self.names, self.mover = bits, names, mover

    def ask(self, parts, instruction, n, seed, group, seen):
        """One request for n answers, each with one code-only retry if it ran out of room and one repair
        attempt if its code cannot run (CP1's ask). Returns the arms."""
        conv = self.prompts.conversation(parts, instruction)
        thinking = self.settings["thinking"]
        outs = self.client.complete(conv, n=n, max_tokens=self.settings["answer_tokens_offline"], thinking=thinking,
                                    seed=seed, **SAMPLING[thinking])
        with self.client.lock:
            took = next((r.get("seconds", 0.0) for r in self.client.cache.values() if r["outputs"] is outs),
                        0.0) / max(n, 1)

        def one(item):
            k, o = item
            tokens, finished, retried = o["tokens"], o["finish"] == "stop", False
            code = extract_code(o["text"])
            arm = (evaluate(code, group, self.ev, self.ev_path) if code else
                   {"code": None, "group": group, "stage": "no-code",
                    "reason": "no ```python block in the answer" + ("" if finished else " (ran out of tokens)")})
            retry = ((code is None and not finished and self.settings["code_only_retry"])
                     or (arm["stage"] in BROKEN and self.settings["repairs"] > 0))
            if retry:
                msg = FINISH if code is None else REPAIR.format(reason=arm["reason"])
                o2 = self.client.complete(conv + [{"role": "assistant", "content": tail(o["text"])},
                                                  {"role": "user", "content": msg}],
                                          n=1, max_tokens=MAX_TOKENS[False], thinking=False, seed=seed + 50,
                                          **SAMPLING[False])[0]
                tokens, retried = tokens + o2["tokens"], True
                code2 = extract_code(o2["text"])
                if code2:
                    arm = evaluate(code2, group, self.ev, self.ev_path)
            arm.update(tokens=tokens, finished=finished, retried=retried, k=k, llm_seconds=round(took, 1))
            return arm
        with ThreadPoolExecutor(max(1, min(n, 4))) as ex:
            arms = list(ex.map(one, enumerate(outs)))
        for arm in arms:                                           # the same code twice is one rule
            arm["duplicate"] = bool(arm["code"] and arm["code"] in seen)
            if arm["code"]:
                seen.add(arm["code"])
        return arms

    def guard(self, parent):
        """Step 4 of the module doc: the guard fitter on a near-miss. Returns the narrowed rule's arm or None."""
        res = parent["res"]
        c = {"id": parent["order"], "group": parent["group"], "code": parent["code"],
             "binding": parent["binding"], "order": parent["order"]}
        rec, made = s3_guards.fit_rule(c, res["applies"], res["claimed_ok"], self.ev.actions, self.bits, self.names,
                                       self.mover, dict(s3_guards.DEFAULTS, auto_exception=False))
        parent["guard_fit"] = {k: rec.get(k) for k in ("guard", "words", "why", "right", "wrong")}
        if not made:
            return None
        arm = evaluate_bound(made[0]["code"], parent["group"], self.ev, self.ev_path, made[0]["binding"])
        arm.update(guard={"words": rec["words"], "literals": rec["guard"]}, duplicate=False)
        return arm

    def run_group(self, group):
        """Rounds for one action group (CP1's run_group, with the guard step). Returns its record."""
        t0 = time.time()
        game, ev, s = self.game, self.ev, self.settings
        seen, arms, steps = set(), [], []

        def add(new, **kw):
            for a in new:
                a.update(order=len(arms), refined=0, **kw)
                arms.append(a)

        parts, instruction = self.prompts.first_round(group, rng_for(game, group, "prompt"))
        add(self.ask(parts, instruction, s["k_first"], 1, group, seen), origin="first round", step=0)
        rng = rng_for(game, group, "bandit")
        stopped, guarded = None, 0
        while stopped is None:
            live = [a for a in arms if not a["duplicate"]]
            book = by_merit([a for a in live if carries(a)])
            left_calls = s["calls_per_group"] - len(steps)
            if book:                                               # the group has a rule that carries: residual
                comp = K.compose([member(a, self.ev_path) for a in book], self.ev_path, select=True)
                left = K.unexplained(comp, ev, group) if comp["stage"] == "checked" else {}
                if len(left) < K.MIN_NEW_CASES:
                    stopped = f"trusted, with unexplained cells left on {len(left)} cases (fewer than {K.MIN_NEW_CASES})"
                    break
                if not left_calls:
                    stopped = f"out of calls; trusted, with unexplained cells left on {len(left)} cases"
                    break
                parts, instruction, shown = self.prompts.residual(group, [book[i] for i in comp["book"]], left,
                                                                  rng_for(game, group, "residual", len(steps) + 1))
                kind, parent = "residual", None
            elif not left_calls:
                stopped = "out of calls; no trusted rule" + (" without level facts" if any(is_trusted(a) for a in live)
                                                             else "")
                break
            else:
                pool = [a for a in live if in_pool(a)] or [a for a in live if a["stage"] == "checked" and
                                                           a["trust"]["moves"]]
                if pool:
                    parent = pool[pick(pool, rng)]
                    if (parent["trust"]["claimed_exactness"] >= NEAR_MISS and not parent["trust"]["trusted"]
                            and "guard_fit" not in parent):
                        narrowed = self.guard(parent)                # no call spent on this
                        if narrowed is not None:
                            add([narrowed], origin="guard", step=None, parent=parent["order"])
                            guarded += 1
                            self.log(f"[{game} {group}] guard on rule {parent['order']}: {narrowed['guard']['words']}"
                                     f" -> {describe_arm(narrowed)}")
                            if carries(narrowed):
                                continue
                    parts, instruction, shown = self.prompts.refinement(
                        group, parent, rng_for(game, group, "prompt"), rng_for(game, group, "cases", len(steps) + 1))
                    parent["refined"] += 1
                    kind = "refine"
                else:                                              # nothing has run yet: ask again, with the reasons
                    parts, instruction = self.prompts.retry(group, [a["reason"] for a in arms],
                                                            rng_for(game, group, "prompt"))
                    kind, parent, shown = "retry", None, 0
            step = len(steps) + 1
            child = self.ask(parts, instruction, 1, 100 + step, group, seen)[0]
            add([child], origin=kind, step=step, parent=None if parent is None else parent["order"], shown=shown)
            steps.append(child)
            self.log(f"[{game} {group}] call {step} ({kind}" + (f" of rule {parent['order']}" if parent is not None
                                                                 else "") + f"): {describe_arm(child)}")
        keys = ev.keys_of(group)
        return {"group": group, "moves": int(ev.count[keys].sum()), "cases": int(len(keys)),
                "changing_cases": int((ev.changed[keys] & ~ev.frozen[keys]).sum()),
                "conflict_cases": int((ev.n_variants[keys] > 1).sum()), "calls": len(steps), "guards": guarded,
                "stopped": stopped, "trusted": sum(is_trusted(a) and not a["duplicate"] for a in arms),
                "carrying": sum(carries(a) and not a["duplicate"] for a in arms),
                "seconds": round(time.time() - t0), "arms": arms}


def in_pool(arm):
    """Worth refining: it runs, covers 20 moves and gets more changes right than wrong."""
    return (arm["stage"] == "checked" and arm["trust"]["moves"] >= ADMIT["min_moves"]
            and arm["trust"]["gain_cells"] > 0)


def is_trusted(arm):
    return arm["stage"] == "checked" and arm["trust"]["trusted"]


def carries(arm):
    """Trusted and free of level facts: the rule a group's search stops at."""
    return is_trusted(arm) and not arm["lint"]


def pick(pool, rng):
    """Thompson sampling, as in CP0: the arm with the highest Beta draw."""
    draws = [rng.beta(1 + BANDIT_C * a["trust"]["claimed_exactness"],
                      1 + BANDIT_C * (1 - a["trust"]["claimed_exactness"]) + a["refined"]) for a in pool]
    return int(np.argmax(draws))


def member(arm, layout_path):
    return dict(src=arm["code"], binding=arm["binding"], layout_path=layout_path, group=arm["group"])


def by_merit(arms):
    """Best first for the book: rules without level facts, then by gain."""
    return sorted(arms, key=lambda a: (bool(a["lint"]), -a["trust"]["gain_cells"], a["order"]))


def describe_arm(a):
    if a["stage"] != "checked":
        return f"{a['stage']} ({(a.get('reason') or '')[:70]})"
    t = a["trust"]
    return (f"{t['claimed_exactness']:.1%} claimed-exact on {t['moves']} moves, {t['cells_right']} cells right"
            + (", TRUSTED" if t["trusted"] else "") + (f", {len(a['lint'])} lint warnings" if a["lint"] else "")
            + f"; {a.get('tokens', 0)} tokens")


# =============================================================================================
# 5. A run: every game's groups in one pool (CP1's run_all)
# =============================================================================================
def level_inputs(run, game):
    """(training split path, its evidence, S1's text per case, feature bits, names, mover) for one game."""
    ev_dir, s1 = os.path.join(run.dir, "evidence"), os.path.join(run.dir, "s1_see")
    p_train = next(os.path.join(ev_dir, f) for f in sorted(os.listdir(ev_dir))
                   if f.startswith(f"{game}_L") and f.endswith("_train.npz"))
    name = os.path.basename(p_train)[:-4]
    with open(os.path.join(s1, name + ".facts.jsonl")) as f:
        texts = {r["k"]: r["text"] for r in map(json.loads, f)}
    facts = np.load(os.path.join(s1, name + ".facts.npz"))
    names = [str(x) for x in facts["feature_names"]]
    bits = np.unpackbits(facts["features"], axis=1)[:, :len(names)].astype(bool)
    with open(os.path.join(s1, name + ".roles.json")) as f:
        mover = json.load(f)["mover"]
    return p_train, LevelEvidence.load(p_train), texts, bits, names, mover


def groups_to_call(ev, settings):
    """The groups that get calls: the level's groups (by moves, at most max_groups) whose moves ever change
    the board outside a frozen stretch (CP1's)."""
    groups = ev.groups[:settings["max_groups"]]
    return [g for g in groups if (ev.changed[ev.keys_of(g)] & ~ev.frozen[ev.keys_of(g)]).any()]


def run_all(run, games, client, log=print, vocab=None):
    """S2 for every game of `run`, with all their groups in one pool of GROUP_WORKERS (module doc)."""
    settings = dict(DEFAULTS, **{k: v for k, v in run.config.get("write", {}).items() if k in DEFAULTS})
    if settings["exceptions"]:
        raise NotImplementedError("exception prompts are not built (the plan's cut line): set "
                                  "[write] exceptions = false")
    vocab = run.config.get("vocab", {}).get("enabled", True) if vocab is None else vocab
    writers, tasks = {}, []
    for game in games:
        p_train, ev, texts, bits, names, mover = level_inputs(run, game)
        writers[game] = Writer(game, ev, p_train, client, settings, texts, bits, names, mover, vocab, log)
        groups = groups_to_call(ev, settings)
        log(f"[{game}] training level {ev.level + 1}: {len(ev.actions)} cases; {len(groups)} groups get calls")
        tasks += [(game, g) for g in groups]
    t0 = time.time()

    def one(task):                    # a group that fails is recorded and logged; the others go on
        game, group = task
        try:
            return writers[game].run_group(group)
        except Exception as e:        # noqa: BLE001 - the run is long; the answer cache makes a rerun cheap
            log(f"[{game} {group}] FAILED: {type(e).__name__}: {e}")
            return {"group": group, "error": f"{type(e).__name__}: {e}", "calls": 0, "guards": 0,
                    "trusted": 0, "carrying": 0, "arms": []}
    with ThreadPoolExecutor(GROUP_WORKERS) as ex:
        records = list(ex.map(one, tasks))
    out = {}
    for game in games:
        recs = [r for (g, _), r in zip(tasks, records) if g == game]
        out[game] = write_game(run, game, writers[game], recs, settings, vocab, client, time.time() - t0)
    return out


def write_game(run, game, writer, records, settings, vocab, client, seconds):
    """One game's files (module doc). Candidates are numbered across the game, group by group."""
    d = os.path.join(run.stage_dir("s2_write"), game)
    os.makedirs(d, exist_ok=True)
    cands = []
    for r in records:
        for a in r["arms"]:
            c = {k: v for k, v in a.items() if k != "res"}
            c.update(id=len(cands), source="s2_write")
            cands.append(c)
    p_cands, p_groups, p_llm = (os.path.join(d, f) for f in ("candidates.jsonl", "groups.json", "llm.json"))
    with open(p_cands, "w") as f:
        for c in cands:
            f.write(json.dumps(plain(c)) + "\n")
    with open(p_groups, "w") as f:
        json.dump([r["group"] for r in records], f)
    llm = {"calls": sum(r["calls"] for r in records), "guards": sum(r["guards"] for r in records),
           "candidates": len(cands), "tokens": sum(int(c.get("tokens") or 0) for c in cands),
           "no_code": sum(c["stage"] == "no-code" for c in cands),
           "ran_out": sum(1 for c in cands if c.get("finished") is False),
           "with_primitives": sum(bool(c.get("primitives")) for c in cands),
           "groups": [{k: v for k, v in r.items() if k != "arms"} for r in records],
           "model": client.served_model(), "run_seconds": round(seconds), "settings": settings, "vocab": vocab}
    with open(p_llm, "w") as f:
        json.dump(plain(llm), f, indent=1)
    run.write_header("s2_write", game, VERSION, dict(settings, vocab=vocab, model=client.served_model()),
                     {"evidence": writer.ev_path}, {"candidates": p_cands, "groups": p_groups, "llm": p_llm},
                     code=__file__)
    trusted = sum(r["trusted"] for r in records)
    run_log = f"[{game}] S2: {llm['calls']} calls after the first round, {llm['guards']} guard rescues tried, " \
              f"{len(cands)} candidates, {trusted} trusted; {llm['tokens']:,} tokens"
    writer.log(run_log)
    return llm


def dry_run(run, games, log=print):
    """Every group's first-round prompt as text and pictures, with its token estimate; no LLM."""
    settings = dict(DEFAULTS, **{k: v for k, v in run.config.get("write", {}).items() if k in DEFAULTS})
    vocab = run.config.get("vocab", {}).get("enabled", True)
    out = {}
    for game in games:
        p_train, ev, texts, bits, names, mover = level_inputs(run, game)
        prompts = Prompter(ev, texts, settings, vocab)
        d = os.path.join(run.stage_dir("s2_write"), game, "prompts")
        os.makedirs(d, exist_ok=True)
        for g in groups_to_call(ev, settings):
            parts, instruction = prompts.first_round(g, rng_for(game, g, "prompt"))
            conv = prompts.conversation(parts, instruction)
            slug = re.sub(r"[^a-z0-9]+", "_", g.lower()).strip("_")
            n_img = 0
            with open(os.path.join(d, slug + ".txt"), "w") as f:
                f.write("=== SYSTEM ===\n" + conv[0]["content"] + "\n\n=== USER ===\n")
                for p in conv[1]["content"]:
                    if p["type"] == "text":
                        f.write(p["text"] + "\n\n")
                    else:
                        n_img += 1
                        png = os.path.join(d, f"{slug}_{n_img}.png")
                        with open(png, "wb") as g_:
                            g_.write(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))
                        f.write(f"[picture {n_img}: {os.path.basename(png)}]\n\n")
            est = prompts.estimate_tokens(parts)
            out[(game, g)] = {"tokens": est, "pictures": n_img}
            log(f"[{game} {g}] first-round prompt: ~{est:,} tokens (estimate), {n_img} pictures")
    return out
