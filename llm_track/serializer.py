"""serializer.py — C1: a transition becomes a feature record and a short text.

This is the foundation of the LLM track: everything the Judge sees, everything
the scorer g learns from, and everything a generated heuristic may read comes
through here. Nothing downstream can be better than this representation, which
is why Probe D measures it directly.

Self-sufficient by construction. Every field is computed from
`(frame, action_idx, next_frame, level)` — exactly the fields already in the
.npz corpus — so the whole labeling study runs over the 14.1M historical
transitions without playing a single new game, and nothing waits on the object-
perception layer. When that layer lands it adds richer component descriptors
behind this same interface; consumers do not change.

Two representations, one computation:
  * `record`  — a JSON-able dict (versioned) plus a fixed-length numeric vector
                for g and the heuristic sandbox.
  * `text`    — a deterministic ~100-200 token rendering for the LLM. No free
                text is ever generated here: the template is fixed, so a
                prompt-format change is a code change with a version bump.

The "displacement guess" is the cheapest possible object-motion detector: for
each colour, the cells that changed away from it and the cells that changed into
it are compared as shapes, and a match is reported as a translation (see
`_detect_moves`). It recovers real object motion with no object layer — on ls20
it reports "a 25-cell colour-3 shape moved by (0, -5)" — but it is blind to
rotation, scaling, occlusion, and to two same-coloured objects moving
differently. That is acceptable at this stage and is precisely what the object
layer later fixes.
"""
import hashlib

import numpy as np

from .tickers import connected_components

SCHEMA_VERSION = "1.0"
GRID = 64
N_CELLS = GRID * GRID
N_COLORS = 16
MAX_COMPONENTS = 8          # largest non-ticker components kept, by cell count

ACTION_NAMES = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4",
                5: "ACTION5", 6: "ACTION6"}


def decode_action(action_idx):
    """Unified index -> (action_type 1-6, (y, x) or None).

    Contract (eval_common.TransitionLogger): 0-4 = ACTION1-5,
    5 + (64*y + x) = a click at (x, y).
    """
    a = int(action_idx)
    if a < 5:
        return a + 1, None
    c = a - 5
    return 6, (c // GRID, c % GRID)


def _shape_key(cells):
    """Translation-invariant key for a set of cells."""
    y0, x0 = cells[:, 0].min(), cells[:, 1].min()
    return frozenset((int(y - y0), int(x - x0)) for y, x in cells)


def _dominant(values):
    """Most common value in a small int array."""
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[counts.argmax()]), len(vals)


def _detect_moves(frame, next_frame, changed):
    """Rigid translations of same-coloured shapes, detected per COLOUR.

    For colour c, the cells that changed AWAY from c are where a colour-c object
    used to be, and the cells that changed INTO c are where it now is. If those
    two sets are the same shape, the object translated, and the offset between
    their bounding-box origins is the displacement.

    Working per colour rather than per component matters: when an object moves
    by less than its own width, the vacated and arrived cells touch and merge
    into ONE connected component, which a component-pair matcher cannot see.
    That is the common case on ls20 (a 5x10 block sliding one cell).

    Conservative by design: if two same-coloured objects move differently the
    shapes will not match and no move is reported, which is the correct answer
    for a detector that cannot tell them apart.
    """
    moves = []
    for c in range(N_COLORS):
        left = changed & (frame == c)
        arrived = changed & (next_frame == c)
        n = int(left.sum())
        if n == 0 or n != int(arrived.sum()):
            continue
        lc = np.argwhere(left).astype(np.int16)
        ac = np.argwhere(arrived).astype(np.int16)
        if _shape_key(lc) != _shape_key(ac):
            continue
        moves.append({
            "color": c, "size": n,
            "dy": int(ac[:, 0].min() - lc[:, 0].min()),
            "dx": int(ac[:, 1].min() - lc[:, 1].min()),
        })
    return moves


def _describe_components(frame, next_frame, diff, ticker_mask):
    """Non-ticker changed components, largest first, with colour composition."""
    nonticker = diff & ~ticker_mask
    comps = connected_components(nonticker)
    comps.sort(key=len, reverse=True)
    comps = comps[:MAX_COMPONENTS]

    out = []
    for cells in comps:
        ys, xs = cells[:, 0], cells[:, 1]
        before = frame[ys, xs]
        after = next_frame[ys, xs]
        cb, nb = _dominant(before)
        ca, na = _dominant(after)
        # Per-component colour flow: which colours this component lost and
        # gained. Without it a component whose colours merely rearrange reads as
        # "colour 3 -> colour 3", which tells the Judge nothing.
        hb = np.bincount(before, minlength=N_COLORS)
        ha = np.bincount(after, minlength=N_COLORS)
        delta = (ha - hb).astype(int)
        out.append({
            "size": int(len(cells)),
            "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
            "centroid": [round(float(ys.mean()), 1), round(float(xs.mean()), 1)],
            "color_before": cb, "color_after": ca,
            "n_colors_before": nb, "n_colors_after": na,
            "color_delta": {int(i): int(d) for i, d in enumerate(delta) if d},
        })
    return out


def serialize(frame, action_idx, next_frame, ticker_mask, *, level=None,
              level_delta=0, is_level_completing=False, action_num=None,
              game=None):
    """Build the feature record for one transition.

    `ticker_mask` is the (64, 64) bool decorative mask from llm_track.tickers.
    Pass an all-False mask to disable the ticker feature (the ablation).
    """
    frame = np.asarray(frame, dtype=np.uint8)
    next_frame = np.asarray(next_frame, dtype=np.uint8)
    diff = frame != next_frame
    n_changed = int(diff.sum())
    n_ticker = int((diff & ticker_mask).sum())
    n_nonticker = n_changed - n_ticker

    comps = _describe_components(frame, next_frame, diff, ticker_mask)
    moves = _detect_moves(frame, next_frame, diff & ~ticker_mask)
    cb = np.bincount(frame.ravel(), minlength=N_COLORS).astype(int)
    ca = np.bincount(next_frame.ravel(), minlength=N_COLORS).astype(int)
    a_type, click = decode_action(action_idx)

    rec = {
        "schema_version": SCHEMA_VERSION,
        "game": game, "action_num": action_num, "level": level,
        "level_delta": int(level_delta),
        "is_level_completing": bool(is_level_completing),
        "action_type": a_type,
        "click": list(click) if click else None,
        "click_color_before": int(frame[click]) if click else -1,
        "click_color_after": int(next_frame[click]) if click else -1,
        "changed": n_changed > 0,
        "n_changed": n_changed,
        "frac_changed": round(n_changed / N_CELLS, 5),
        "n_changed_ticker": n_ticker,
        "n_changed_nonticker": n_nonticker,
        # The decisive flag: the frame moved, but only decoration moved.
        "ticker_only": n_changed > 0 and n_nonticker == 0,
        "n_components": len(comps),
        "components": comps,
        "moves": moves,
        "color_before": cb.tolist(),
        "color_after": ca.tolist(),
        "color_delta": (ca - cb).tolist(),
        "n_distinct_colors_before": int((cb > 0).sum()),
        "n_distinct_colors_after": int((ca > 0).sum()),
        "background_color": int(cb.argmax()),
    }
    rec["text"] = render_text(rec)
    return rec


def render_text(rec):
    """Deterministic ~100-200 token rendering. Template is fixed on purpose."""
    parts = [ACTION_NAMES[rec["action_type"]]]
    if rec["click"]:
        y, x = rec["click"]
        parts[0] += f" at (row {y}, col {x}), colour {rec['click_color_before']}"
        if rec["click_color_after"] != rec["click_color_before"]:
            parts[0] += f"->{rec['click_color_after']}"
    if not rec["changed"]:
        parts.append("no change")
    else:
        parts.append(f"changed {rec['n_changed']} cells ({100*rec['frac_changed']:.2f}%)")
        if rec["ticker_only"]:
            parts.append("ONLY the known indicator/ticker region changed")
        elif rec["n_changed_ticker"]:
            parts.append(f"{rec['n_changed_ticker']} of them in the ticker region")
        for m in rec["moves"]:
            parts.append(f"a {m['size']}-cell colour-{m['color']} shape MOVED by "
                         f"({m['dy']:+d} row, {m['dx']:+d} col)")
        for i, c in enumerate(rec["components"]):
            h = c["bbox"][2] - c["bbox"][0] + 1
            w = c["bbox"][3] - c["bbox"][1] + 1
            flow = ", ".join(f"colour {k} {d:+d}" for k, d in
                             sorted(c["color_delta"].items(), key=lambda kv: -abs(kv[1]))[:4])
            parts.append(f"comp{i+1}: {c['size']} cells ({h}x{w}) at row "
                         f"{c['bbox'][0]}, col {c['bbox'][1]}"
                         + (f" [{flow}]" if flow else ""))
        deltas = [f"colour {i}: {b}->{a}"
                  for i, (b, a, d) in enumerate(zip(rec["color_before"],
                                                    rec["color_after"],
                                                    rec["color_delta"])) if d]
        if deltas:
            parts.append("counts " + "; ".join(deltas[:6]))
    if rec["is_level_completing"]:
        parts.append("*** THIS ACTION COMPLETED THE LEVEL ***")
    return " | ".join(parts)


# Fields that make two transitions "the same event" for dedupe. Deliberately
# excludes action_num and level: the same tick at action 900 and action 90,000
# is one signature with multiplicity 2, which is what makes labeling affordable.
def signature(rec):
    """Stable hash of a record's structural content (for dedupe)."""
    comps = [(c["size"], tuple(c["bbox"]), c["color_before"], c["color_after"])
             for c in rec["components"]]
    key = (rec["schema_version"], rec["game"], rec["action_type"],
           tuple(rec["click"] or ()), rec["click_color_before"],
           rec["click_color_after"], rec["n_changed"], rec["n_changed_ticker"],
           tuple(comps), tuple(sorted((m["color"], m["dy"], m["dx"]) for m in rec["moves"])),
           tuple(rec["color_delta"]))
    return hashlib.blake2b(repr(key).encode(), digest_size=8).hexdigest()


# ---- numeric feature vector (what g and the sandbox consume) ----------------
# Fixed layout, so a trained g artifact is only valid for its SCHEMA_VERSION.
VECTOR_FIELDS = (
    ["action_type", "is_click", "click_y", "click_x", "click_color_before",
     "click_color_after", "changed", "n_changed", "frac_changed",
     "n_changed_ticker", "n_changed_nonticker", "ticker_only", "n_components",
     "n_moves", "max_move_dy", "max_move_dx", "n_distinct_colors_before",
     "n_distinct_colors_after", "background_color"]
    + [f"comp{i}_{f}" for i in range(MAX_COMPONENTS)
       for f in ("size", "y0", "x0", "h", "w", "color_before", "color_after")]
    + [f"color_delta_{c}" for c in range(N_COLORS)]
)
VECTOR_LEN = len(VECTOR_FIELDS)


def to_vector(rec):
    """Fixed-length float32 feature vector. Order is VECTOR_FIELDS."""
    click = rec["click"] or (-1, -1)
    mv_dy = max((abs(m["dy"]) for m in rec["moves"]), default=0)
    mv_dx = max((abs(m["dx"]) for m in rec["moves"]), default=0)
    v = [rec["action_type"], 1.0 if rec["click"] else 0.0, click[0], click[1],
         rec["click_color_before"], rec["click_color_after"],
         1.0 if rec["changed"] else 0.0, rec["n_changed"], rec["frac_changed"],
         rec["n_changed_ticker"], rec["n_changed_nonticker"],
         1.0 if rec["ticker_only"] else 0.0, rec["n_components"],
         len(rec["moves"]), mv_dy, mv_dx, rec["n_distinct_colors_before"],
         rec["n_distinct_colors_after"], rec["background_color"]]
    for i in range(MAX_COMPONENTS):
        if i < len(rec["components"]):
            c = rec["components"][i]
            y0, x0, y1, x1 = c["bbox"]
            v += [c["size"], y0, x0, y1 - y0 + 1, x1 - x0 + 1,
                  c["color_before"], c["color_after"]]
        else:
            v += [0, -1, -1, 0, 0, -1, -1]
    v += rec["color_delta"]
    return np.asarray(v, dtype=np.float32)
