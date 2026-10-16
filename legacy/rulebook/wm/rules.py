"""rules.py - what the LLM is shown and asked when it writes a rule, and how its
answer is read (docs/plans/llm-rulebook.md section 3.2).

The format is Stage A's (legacy/llm_track/rule_writer.py), the only one that has
worked here: for one ACTION GROUP (a button, or clicks on one colour) the model
sees the level's first board, where the ticker is, and up to 6 recorded moves as
before/after pictures with the changes boxed PLUS the exact changed cells as
text (pictures alone made it misread colours and distances). It answers with one
```python block defining RULE, applies(board, act, api), predict(board, act, api).

Round 2 adds the best rules so far with their scores, moves where the best one
was wrong (before | its prediction | the real after) and one move it missed.

Moves shown are picked to cover DISTINCT EFFECTS (most common first), plus up to
two that changed nothing; moves from frozen stretches are never shown.
"""
import base64
import collections
import io
import re

import numpy as np
from PIL import Image, ImageDraw

from .evidence import GRID, PALETTE_NAMES, Act, connected_components

N_EXAMPLES, N_UNCHANGED = 6, 2     # moves shown per group; up to 2 of them unchanged
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

RULE_DOC = """You write ONE rule about how this game works, as Python. `np` (numpy) is \
available; nothing can be imported.

    RULE = "one line saying what happens"
    def applies(board, act, api):
        # True if your rule knows what this action does on this board
        ...
    def predict(board, act, api):
        # the board right AFTER the action
        out = board.copy()
        ...
        return out

`board` is the board BEFORE the action: a 64x64 numpy array of colour indices 0-15.
It is read-only, so work on board.copy().
`act.action` is 1-5 for a button (ACTION1-ACTION5) or 6 for a click; `act.click`
is (row, col) for a click and None otherwise.
`api` gives you:
  api.first       the board at the START of this level (64x64)
  api.background  the most common colour of api.first
  api.ticker      64x64 bool: timer/counter cells that change by themselves. They
                  are ignored when your rule is checked.
  api.layout      regions on api.first: connected same-colour areas (background and
                  ticker excluded), largest first. Each region r has r.colour,
                  r.size, r.y0, r.x0, r.y1, r.x1 (inclusive bounding box), r.mask
                  (64x64 bool, True on the region) and r.cells (n x 2 (row, col)).

How rules are checked: on every recorded move where applies() is True, predict()
must reproduce the real next board EXACTLY on every non-ticker cell. A rule is kept
if it applies to at least 20 recorded moves and is exactly right on at least 95%
of them; it is trusted for planning only if it is right on ALL of them. So:
- make applies() narrow enough that the rule is right whenever it says it applies;
- a move no rule covers is treated as unknown, so a rule is only worth something if
  it predicts the CHANGES exactly;
- it is used on OTHER levels of this game, so describe the mechanism, not
  specific positions;
- keep it fast: whole-array numpy, no Python loops over all 4,096 cells.

Example, from a DIFFERENT game - it shows the style, not the answer here:

    RULE = "ACTION1 moves the green (14) block up by one row"
    def applies(board, act, api):
        return act.action == 1 and (board == 14).any()
    def predict(board, act, api):
        out = board.copy()
        g = board == 14
        out[g] = api.background
        out[np.roll(g, -1, axis=0)] = 14
        return out"""

LEGEND = ", ".join(f"{i} {n}" for i, n in enumerate(PALETTE_NAMES))
SYSTEM = ("You work out the exact mechanics of an unknown grid game from recorded moves, "
          "and write each one down as a small Python rule.\n\n" + RULE_DOC +
          f"\n\nColour numbers are drawn as: {LEGEND}.")
TASK = ("Write ONE rule for this action group: what exactly does this action do to the board? "
        "Your rule is checked on ALL recorded moves of this level, not only the ones shown, so "
        "applies() must return False for moves it does not describe. It is also used on other "
        "levels of this game, so describe the mechanism, not these particular positions.")
ANSWER_FORMAT = {
    False: "First describe in at most 5 sentences exactly what this action does: which cells "
           "change, to which colours, and when nothing changes. Then give exactly one ```python "
           "code block that defines RULE, applies and predict.",
    True: "Think it through, but keep it brief: an answer that runs out of room before the "
          "code is wasted. Then give exactly one ```python code block that defines RULE, "
          "applies and predict.",
}
REPAIR = ("Your rule could not be used: {reason}\nFix it and answer with exactly one ```python "
          "code block that defines RULE, applies and predict.")
FINISH = ("You ran out of room before writing the code. Write it now, with no further analysis: "
          "exactly one ```python code block that defines RULE, applies and predict.")


# --------------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------
# Describing one recorded move exactly
# --------------------------------------------------------------------------------
def _shape_key(cells):
    y0, x0 = cells[:, 0].min(), cells[:, 1].min()
    return frozenset((int(y - y0), int(x - x0)) for y, x in cells)


def detect_moves(before, after, changed):
    """Rigid translations of same-coloured shapes, per colour (copied from
    legacy/llm_track/serializer.py): [(colour, size, drow, dcol)]."""
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
    """What kind of change a move made, position-free: used to show the model
    different effects rather than six copies of the commonest one."""
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


# --------------------------------------------------------------------------------
# The prompts
# --------------------------------------------------------------------------------
def pick_examples(ev, group, rng):
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
    want, pick, depth = min(N_EXAMPLES - n_unch, len(changing)), [], 0
    while len(pick) < want:                                   # round-robin over effects
        for s in sigs:
            if depth < len(by_sig[s]) and len(pick) < want:
                pick.append(int(by_sig[s][depth]))
        depth += 1
    pick += [int(unchanged[j]) for j in rng.permutation(len(unchanged))[:n_unch]]
    return [pick[j] for j in rng.permutation(len(pick))]


def group_prompt(ev, group, rng):
    """What round 1 shows for one action group (content parts)."""
    keys = ev.keys_of(group)
    moves = int(ev.count[keys].sum())
    ch_moves = int(ev.count[keys][ev.changed[keys]].sum())
    sizes = [int(((ev.before[k] != ev.after[k]) & ev.live).sum()) for k in keys if ev.changed[k]]
    stats = (f"The recordings have {moves} moves in this group on {len(keys)} different boards; the "
             f"board changed (outside the ticker) on {ch_moves / max(moves, 1):.0%} of the moves"
             + (f", median {int(np.median(sizes))} cells changed." if sizes else "."))
    parts = [text_part("The board at the start of this level:"),
             image_part(render_board(ev.first)),
             text_part(f"{ticker_line(ev.mask)}\n\nACTION GROUP: {group}.\n{stats}\n\nRecorded moves "
                       "of this group. Each picture shows the board BEFORE the move (left) and AFTER it "
                       "(right); a cyan box marks what changed.")]
    for j, k in enumerate(pick_examples(ev, group, rng), 1):
        parts += [text_part(f"Move {j}: {move_text(ev, k)}"),
                  image_part(render_move(ev.before[k], ev.after[k], ev.mask))]
    return parts


def _rank(c):
    g = c["grade"]
    return (g["plan_eligible"], g["gain"], g["admitted"], g["accuracy"])


def feedback(cands, ev, group, rng):
    """What round 2 adds for one group: the best rules so far and their mistakes."""
    mine = [c for c in cands if c["group"] == group]
    checked = sorted((c for c in mine if c["stage"] == "checked"), key=_rank, reverse=True)[:3]
    lines = ["Rules tried so far for this group, best first:"]
    for c in checked:
        g = c["grade"]
        verdict = (" - TRUSTED (right on every move)." if g["plan_eligible"] or g["known_noop"] else
                   " - kept, but not right on every move." if g["admitted"] else
                   f" - not kept: {g['reason']}.")
        lines += [f"\nRULE: {c['rule']}\nIt applies to {g['moves']} recorded moves "
                  f"({g['in_group_moves']} of this group's {g['group_moves']}), is exactly right on "
                  f"{g['accuracy']:.0%} of them, and predicts {g['changes_right']} real changes exactly"
                  + verdict, "```python", c["code"], "```"]
    if not checked:
        lines.append("\n(none has run successfully yet)")
    parts = [text_part("\n".join(lines))]
    if checked:
        best = checked[0]
        wrong = best["res"]["wrong"]
        for j in rng.permutation(len(wrong))[:2]:
            k, pred = wrong[j]
            parts += [text_part(
                f"The best rule was WRONG on this move: {move_text(ev, k)}\nWhere its prediction "
                "differs from the real board:\n  " + "\n  ".join(
                    _areas(pred, ev.after[k], ev.mask, lambda p, q, n: f"predicted {p}, really {q} x{n}", 12))
                + "\nPicture: BEFORE | THE RULE'S PREDICTION | REAL AFTER (cyan boxes: where the "
                "prediction is wrong)."),
                image_part(render_compare(ev.before[k], pred, ev.after[k], ev.mask))]
        missed = best["grade"].get("missed") or []
        if missed:
            k = int(rng.choice(missed))
            parts += [text_part("The best rule did not apply to this move of the group, where the "
                                f"board DID change: {move_text(ev, k)}"),
                      image_part(render_move(ev.before[k], ev.after[k], ev.mask))]
    rej = collections.Counter(c["reason"][:90] for c in mine if c["stage"] not in ("checked", "duplicate"))
    tail = (["\nSome candidates could not be used:"] + [f"  {n} x {why}" for why, n in rej.most_common(4)]) if rej else []
    tail.append("\nWrite ONE new rule for this group that is exactly right on every move it covers, "
                "or covers more of the group's moves: improve the best one or try a different idea.")
    parts.append(text_part("\n".join(tail)))
    return parts


def conversation(parts, thinking=True):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": list(parts) + [text_part(TASK + "\n\n" + ANSWER_FORMAT[thinking])]}]


# --------------------------------------------------------------------------------
# Reading an answer
# --------------------------------------------------------------------------------
def extract_code(text):
    """The last ```python block of the final answer (after any thinking)."""
    answer = (text or "").split("</think>")[-1]
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)```", answer, re.S)
    return blocks[-1].strip() if blocks else None


def tail(text, limit=24000):
    """What a retry is shown of the first answer. An answer that ran out of room
    is ALL thinking (nothing after </think>, or no </think> at all), and that
    reasoning is exactly what the retry needs, so it is kept; only its last
    `limit` characters, because 20k tokens of thinking would not fit beside a
    fresh reply."""
    think, _, answer = (text or "").rpartition("</think>")
    answer = answer.strip() or think.strip()
    return answer if len(answer) <= limit else "[...earlier reasoning omitted...]\n" + answer[-limit:]
