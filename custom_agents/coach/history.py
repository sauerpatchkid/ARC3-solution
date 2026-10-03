"""history.py - what the coach knows about the current level, and the text it
shows the LLM (docs/plans/llm-coach.md section 4.2).

`History.observe(frame, action_idx, next_frame, level)` is called once per
transition, in order. The live agent and the offline probe (which replays a
recorded corpus) feed it exactly the same way, so the probe sees what the online
coach would have seen at that move. `History.render()` returns the summary text
plus the objects it names, so an answer's object ids can be mapped back to
screen cells.

Conventions (the repo's): frames are 64x64 uint8 colour indices; action_idx is
the unified index, 0-4 = ACTION1-5 and 5 + 64*y + x = a click at column x,
row y; `level` is the score when the move was made, so the move that completes
a level is logged with the old score and the next move with the new one.

Choices:
  - Decorations: its own BarDetector (custom_agents/upgrades.py), the sticky
    progress-bar mask mb_gated_att uses. Masked cells never count as changes.
  - Objects: upgrades.screen_objects (single-colour connected regions, most
    salient first: small and rare-coloured). Clicks are credited to the object
    under the clicked cell, keyed by (colour, bounding box), so an object that
    moves becomes a new entry - the honest reading for a summary.
  - "New screen": the masked screen was never seen before in this level. When
    the sticky mask grows, older keys are not re-keyed; this only affects the
    first few thousand moves of a level, where the mask is still settling.
  - Effects are described in words the LLM can use: "no-op", "colour 3 moved
    by (+5, +0)", "colour 9 -> 12"; each object and button shows its most
    common effect with a count ("colour 9 -> 12 (812 of 900)").
  - The previous levels' wins are kept: the winning move and how the colour
    counts changed between the level's first screen and the winning screen.
Deterministic: the same transitions give the same text, byte for byte.
"""
from collections import Counter, OrderedDict, deque

import numpy as np
import xxhash

from upgrades import BarDetector, MAX_OBJECT_CELLS, label_components, screen_objects

SUMMARY_VERSION = "coach-summary-v1"
GRID = 64
N_COLOURS = 16
N_OBJ = 30                  # objects shown, most salient first (+ any with click history)
BUTTON_WINDOW = 200         # last N uses per button
MAX_WINS = 3                # previous levels shown
MAX_CHARS = 5600            # ~2k tokens at ~2.8 chars/token for this table-heavy text
BUTTON_NAMES = [f"ACTION{k}" for k in range(1, 6)]


# --------------------------------------------------------------------------------
# Effects: what one transition did, in words
# --------------------------------------------------------------------------------
def _shape_key(cells):
    """Translation-invariant key for a set of (row, col) cells."""
    y0, x0 = cells[:, 0].min(), cells[:, 1].min()
    return frozenset((int(y - y0), int(x - x0)) for y, x in cells)


def detect_moves(frame, next_frame, changed):
    """Rigid translations of same-coloured shapes, per colour. Copied from
    legacy/llm_track/serializer.py (_detect_moves; legacy/ is a leaf and is
    never imported): for colour c, the cells that changed away from c and the
    cells that changed into c are the same shape when a colour-c object slid.
    Returns [(colour, size, dy, dx)]."""
    out = []
    involved = np.union1d(frame[changed], next_frame[changed])     # colours that took part
    for c in involved.tolist():
        left = changed & (frame == c)
        arrived = changed & (next_frame == c)
        n = int(left.sum())
        if n == 0 or n != int(arrived.sum()):
            continue
        lc, ac = np.argwhere(left), np.argwhere(arrived)
        if _shape_key(lc) != _shape_key(ac):
            continue
        out.append((c, n, int(ac[:, 0].min() - lc[:, 0].min()), int(ac[:, 1].min() - lc[:, 1].min())))
    return out


def describe(frame, next_frame, changed):
    """(pattern, detail) for one transition. `pattern` drops cell counts so
    repeated effects of one button group together; `detail` keeps them."""
    n = int(changed.sum())
    if n == 0:
        return "no-op", "no-op"
    moves = detect_moves(frame, next_frame, changed)
    if moves:
        # When a block slides over a floor or background, the floor colour
        # "slides" the opposite way too. The block is the rarer colour on the
        # screen, so report the move of the rarest colour (ties: bigger, lower).
        total = np.bincount(frame.ravel(), minlength=N_COLOURS)
        c, size, dy, dx = min(moves, key=lambda m: (int(total[m[0]]), -m[1], m[0]))
        # (dx, dy): x is the column, matching the object boxes below.
        return (f"colour {c} moved by ({dx:+d}, {dy:+d})",
                f"colour {c} ({size} cells) moved by ({dx:+d}, {dy:+d})")
    before = np.bincount(frame[changed], minlength=N_COLOURS)
    after = np.bincount(next_frame[changed], minlength=N_COLOURS)
    lost = [int(c) for c in np.argsort(-(before - after)) if before[c] > after[c]][:2]
    gained = [int(c) for c in np.argsort(-(after - before)) if after[c] > before[c]][:2]
    flow = f"colour {'/'.join(map(str, lost)) or '?'} -> {'/'.join(map(str, gained)) or '?'}"
    return flow, f"{n} cells changed: {flow}"


# --------------------------------------------------------------------------------
# Objects on one screen
# --------------------------------------------------------------------------------
class Objects:
    """The objects on one masked screen. Crediting a click only needs the
    object under the cursor, so that is measured on demand; the full ranked
    list (upgrades.screen_objects order: small, rare-coloured first) is built
    only when a summary is rendered. Both use screen_objects' rule: background
    = the most common unmasked colour, regions over MAX_OBJECT_CELLS are
    background too, so the labels and keys agree."""

    def __init__(self, frame, mask):
        self.frame, self.mask = frame, mask
        colours, counts = np.unique(frame[~mask], return_counts=True)
        background = colours[np.argmax(counts)] if len(colours) else -1
        self.labels = label_components(frame, (~mask) & (frame != background))
        self._info = {}
        self._items = None

    def _measure(self, lab):
        ys, xs = np.nonzero(self.labels == lab)
        if len(ys) == 0 or len(ys) > MAX_OBJECT_CELLS:
            return None
        return {"label": int(lab), "colour": int(self.frame[ys[0], xs[0]]),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),   # x0, y0, x1, y1
                "size": int(len(ys)),
                # translation-invariant shape: cell offsets from the box corner, row-major
                "shape": xxhash.xxh64(((ys - ys.min()) * GRID + (xs - xs.min())).astype(np.int32).tobytes()).intdigest()}

    def at(self, y, x):
        """The object under cell (y, x), or None for background/decoration."""
        lab = int(self.labels[y, x])
        if lab < 0:
            return None
        if lab not in self._info:
            self._info[lab] = self._measure(lab)
        return self._info[lab]

    @property
    def items(self):
        if self._items is None:
            _, ranked = screen_objects(self.frame, self.mask)
            items = []
            for lab, target in ranked:
                if lab not in self._info:
                    self._info[lab] = self._measure(lab)
                it = dict(self._info[lab], target=int(target))
                items.append(it)
            shapes = Counter(it["shape"] for it in items)
            for it in items:
                it["copies"] = shapes[it["shape"]] - 1
            self._items = items
        return self._items


def obj_key(it):
    return (it["colour"], it["bbox"])


# --------------------------------------------------------------------------------
# One level's record
# --------------------------------------------------------------------------------
class Level:
    def __init__(self, index, first_frame):
        self.index = index
        self.first_frame = first_frame
        self.moves = 0
        self.seen = set()
        self.last_new = 0                          # move number of the last new screen
        self.clicks = {}                           # obj_key -> [tried, changed, new, Counter(pattern)]
        self.background = [0, 0, 0]                # clicks off any object: tried, changed, new
        self.buttons = [deque(maxlen=BUTTON_WINDOW) for _ in range(5)]   # (changed, new, pattern)


class History:
    def __init__(self):
        self.bars = BarDetector()
        self.level = None
        self.wins = []                              # completed levels, oldest first
        self.used = set()                           # action types used this run (0-4, 5 = click)
        self._objects = OrderedDict()               # (frame hash, mask growths) -> Objects
        self._last = None                           # the previous transition

    # -- feeding ---------------------------------------------------------------
    def observe(self, frame, action_idx, next_frame, level):
        frame = np.asarray(frame, dtype=np.uint8)
        next_frame = np.asarray(next_frame, dtype=np.uint8)
        level = int(level)
        if self.level is None:
            self._open_level(level, frame)
        elif level != self.level.index:
            self._close_level(level, frame)
        self.bars.update(frame, next_frame)
        mask = self.bars.sticky
        lv = self.level
        lv.moves += 1
        changed = (frame != next_frame) & ~mask
        key = self._key(next_frame)
        new = key not in lv.seen
        if new:
            lv.seen.add(key)
            lv.last_new = lv.moves
        pattern, _ = describe(frame, next_frame, changed)
        did = bool(changed.any())
        a = int(action_idx)
        if a < 5:
            self.used.add(a)
            lv.buttons[a].append((did, new, pattern))
        else:
            self.used.add(5)
            y, x = divmod(a - 5, GRID)
            it = self.objects(frame).at(y, x)
            if it is None:
                b = lv.background
                b[0] += 1; b[1] += did; b[2] += new
            else:
                rec = lv.clicks.setdefault(obj_key(it), [0, 0, 0, Counter()])
                rec[0] += 1; rec[1] += did; rec[2] += new; rec[3][pattern] += 1
        self._last = (frame, a, next_frame)

    def _close_level(self, new_index, first_frame):
        """The level changed: the previous transition was the winning move."""
        lv = self.level
        if self._last is not None and new_index > lv.index:
            win_frame, a, _ = self._last
            if a < 5:
                move = f"pressing {BUTTON_NAMES[a]}"
            else:
                y, x = divmod(a - 5, GRID)
                it = self.objects(win_frame).at(y, x)
                move = (f"clicking a colour-{it['colour']} object ({it['size']} cells, box "
                        f"{it['bbox'][0]},{it['bbox'][1]}-{it['bbox'][2]},{it['bbox'][3]})"
                        if it else f"clicking empty cell ({x},{y})")
            m = self.bars.sticky
            c0 = np.bincount(lv.first_frame[~m], minlength=N_COLOURS)
            c1 = np.bincount(win_frame[~m], minlength=N_COLOURS)
            d = [(abs(int(c1[c]) - int(c0[c])), c) for c in range(N_COLOURS) if c1[c] != c0[c]]
            flows = [f"colour {c} {int(c0[c])}->{int(c1[c])}" for _, c in sorted(d, reverse=True)[:4]]
            self.wins.append({"level": lv.index, "moves": lv.moves, "move": move,
                              "counts": ", ".join(flows) or "no colour count changed"})
        self._open_level(new_index, first_frame)

    def _key(self, frame):
        return xxhash.xxh64(np.where(self.bars.sticky, 0, frame).tobytes()).intdigest()

    def _open_level(self, index, first_frame):
        """A level starts on its first screen, which counts as seen (a reset
        returns there without a logged transition reaching it)."""
        self.level = Level(index, first_frame)
        self.level.seen.add(self._key(first_frame))

    def objects(self, frame):
        """Objects on a frame under the current mask, cached (screens repeat)."""
        k = (xxhash.xxh64(frame.tobytes()).intdigest(), self.bars.growths)
        hit = self._objects.get(k)
        if hit is None:
            hit = Objects(frame, self.bars.sticky)
            self._objects[k] = hit
            if len(self._objects) > 2048:
                self._objects.popitem(last=False)
        else:
            self._objects.move_to_end(k)
        return hit

    # -- the text --------------------------------------------------------------
    def render(self, frame, available=None, max_chars=MAX_CHARS):
        """The summary for the screen `frame` (the one the next move is made
        on). `available`: action types the game offers now (0-4 buttons, 5 =
        clicks); defaults to the types used so far this run, the best the
        offline probe can do. Returns {"text", "objects", "version"}, where
        objects[i] is the object the text calls id i (with its target cell)."""
        frame = np.asarray(frame, dtype=np.uint8)
        lv = self.level or Level(0, frame)
        avail = sorted(self.used if available is None else set(available))
        objs = self.objects(frame).items
        shown = [it for it in objs[:N_OBJ]] + [it for it in objs[N_OBJ:] if obj_key(it) in lv.clicks]
        while True:
            text = self._text(lv, frame, avail, shown)
            if len(text) <= max_chars or not shown:
                break
            drop = next((i for i in range(len(shown) - 1, -1, -1) if obj_key(shown[i]) not in lv.clicks),
                        len(shown) - 1)
            shown = shown[:drop] + shown[drop + 1:]
        return {"text": text, "version": SUMMARY_VERSION,
                "objects": [{"id": i, "colour": it["colour"], "bbox": it["bbox"], "size": it["size"],
                             "target": it["target"], "label": it["label"]} for i, it in enumerate(shown)]}

    def _text(self, lv, frame, avail, shown):
        names = [BUTTON_NAMES[a] for a in avail if a < 5] + (["clicks"] if 5 in avail else [])
        out = [f"LEVEL {lv.index + 1}. Moves in this level: {lv.moves:,}. "
               f"Different screens seen: {len(lv.seen):,}. "
               f"Moves since the last new screen: {lv.moves - lv.last_new:,}.",
               f"Available actions: {', '.join(names) or 'unknown'}.", ""]
        if 5 in avail:
            out.append("OBJECTS ON SCREEN (box = x0,y0-x1,y1; x is the column, y the row)")
            out.append("id | colour | box | cells | same-shape copies | clicks tried | changed the screen "
                       "| reached a new screen | most common effect")
            for i, it in enumerate(shown):
                t, c, n, pats = lv.clicks.get(obj_key(it), (0, 0, 0, None))
                x0, y0, x1, y1 = it["bbox"]
                if pats:
                    top, k = pats.most_common(1)[0]
                    eff = f"{top} ({k} of {t})"
                else:
                    eff = "never clicked"
                out.append(f"{i} | {it['colour']} | {x0},{y0}-{x1},{y1} | {it['size']} | {it['copies']} | "
                           f"{t} | {c} | {n} | {eff}")
            if not shown:
                out.append("(no objects: the screen is background only)")
            b = lv.background
            out.append(f"Clicks on empty background: {b[0]} tried, {b[1]} changed the screen, "
                       f"{b[2]} reached a new screen.")
            out.append("")
        buttons = [a for a in avail if a < 5]
        if buttons:
            out.append(f"BUTTONS (last {BUTTON_WINDOW} uses in this level)")
            out.append("button | uses | changed the screen | reached a new screen | most common effect")
            for a in buttons:
                h = lv.buttons[a]
                if not h:
                    out.append(f"{BUTTON_NAMES[a]} | 0 | - | - | never pressed")
                    continue
                n = len(h)
                pats = Counter(p for _, _, p in h)
                top, cnt = pats.most_common(1)[0]
                out.append(f"{BUTTON_NAMES[a]} | {n} | {100 * sum(d for d, _, _ in h) // n}% | "
                           f"{100 * sum(w for _, w, _ in h) // n}% | {top} ({cnt} of {n})")
            out.append("")
        if self.wins:
            out.append("PREVIOUS LEVELS")
            for w in self.wins[-MAX_WINS:]:
                out.append(f"Level {w['level'] + 1} was won after {w['moves']:,} moves by {w['move']}. "
                           f"From its first screen to the winning screen: {w['counts']}.")
        return "\n".join(out).rstrip() + "\n"
