"""probe_images.py — render each probe move as a picture, for the image judge.

Probe A (text) was NO-GO, and the post-hoc check located the reason: the
progress signal is in the board state, which a one-line description of the
change omits. The design's planned fallback is to show the judge the board.
This script renders one image per move and writes images.json beside them;
`judge.py --images` attaches them. pairs.jsonl is untouched, so the image run
answers the SAME questions as the text run.

ONE IMAGE PER MOVE: the whole board before the move (left) and after it
(right), in the game's real palette (the one custom_agents/view_utils.py uses),
with a cyan box around each changed non-ticker region, so a small model does
not have to find a 36-cell change on a 4,096-cell board unaided. Cyan is not in
the game's palette, so the box can never be mistaken for game content.

SIZE: Qwen3.5's vision encoder uses 16 px patches merged 2x2, so one visual
token covers 32x32 px. At 8 px per cell each 64x64 board is 512x512 px and a
move image is 1056x512 = 33 x 16 = 528 tokens: every cell is half a patch,
enough to resolve the one-cell detail inside ft09's example symbols.

VERIFIED AGAINST THE PAIRS: every rendered transition is re-diffed and its
changed-cell counts compared with the pair record; any mismatch aborts, so an
image can never silently show a different move from the one the text describes.

    uv run python -m llm_track.probe_images    # -> results/llm/probeA/images/
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from .corpus import GOOSE, CorpusReader, find_corpora, game_of, largest_per_game
from .probe_pairs import build_masks
from .serializer import MAX_COMPONENTS
from .tickers import connected_components

# The game's palette, copied from custom_agents/view_utils.py (not imported:
# llm_track must not depend on the agent package's plotting code).
PALETTE_HEX = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
               "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
               "#FF851B", "#921231", "#4FCC30", "#A356D6"]
PALETTE_NAMES = ["white", "light gray", "gray", "dark gray", "darker gray", "black",
                 "pink", "light pink", "red", "blue", "light blue", "yellow",
                 "orange", "dark red", "green", "purple"]
PAL = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in PALETTE_HEX], np.uint8)

CELL = 8                     # px per grid cell (see SIZE above)
GAP = 32                     # px gutter between the before and after panels
GUTTER = (58, 58, 90)        # dark slate: not a game colour
BOX = (0, 255, 255)          # cyan: not a game colour


def render_board(frame, cell=CELL):
    """64x64 colour indices -> (64*cell)^2 RGB array (512x512 at the default)."""
    rgb = PAL[np.asarray(frame, dtype=np.uint8)]
    return np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)


def render_move(frame, next_frame, mask, cell=CELL):
    """Before | after, with a cyan box around each changed non-ticker region."""
    side, gap = 64 * cell, GAP * cell // CELL
    img = Image.new("RGB", (2 * side + gap, side), GUTTER)
    for k, f in enumerate((frame, next_frame)):
        img.paste(Image.fromarray(render_board(f, cell)), (k * (side + gap), 0))
    draw = ImageDraw.Draw(img)
    comps = sorted(connected_components((frame != next_frame) & ~mask), key=len, reverse=True)
    for c in comps[:MAX_COMPONENTS]:          # the same components the text describes
        y0, x0 = c.min(axis=0)
        y1, x1 = c.max(axis=0)
        for k in (0, 1):
            ox = k * (side + gap)
            draw.rectangle([ox + max(int(x0) * cell - 2, 0), max(int(y0) * cell - 2, 0),
                            ox + min((int(x1) + 1) * cell + 1, side - 1),
                            min((int(y1) + 1) * cell + 1, side - 1)],
                           outline=BOX, width=2)
    return img


def corpus_of(pair, side, results, largest):
    """Where a side's transition lives. Explicit 'corpus' wins; anchors encode
    their run in the event id; sanity pairs came from each game's largest Goose
    run (probe_pairs.sanity_pairs), which largest_per_game reproduces."""
    s = pair[side]
    if "corpus" in s:
        return s["corpus"]
    if pair["tier"] == "anchor":
        run_id, name, _ = pair["event"].split("/")
        return os.path.join(results, "runs", run_id, name, "transitions")
    return largest[pair["game"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="results/llm/probeA/pairs.jsonl")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <pairs dir>/images")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.pairs), "images")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    with open(a.pairs) as f:
        pairs = [json.loads(ln) for ln in f]
    corpora = find_corpora(a.results, agent=GOOSE)
    games = {p["game"] for p in pairs}
    masks, _, _ = build_masks([c for c in corpora if game_of(c[0]) in games])
    largest = largest_per_game(corpora)

    readers, index, done, mapping = {}, {}, {}, {}
    for p in pairs:
        mapping[p["pair_id"]] = {}
        for side in ("x", "y"):
            d = corpus_of(p, side, a.results, largest)
            if d not in readers:
                readers[d] = CorpusReader(d)
                index[d] = {int(n): k for k, n in enumerate(readers[d].scalars["action_nums"])}
            an = p[side]["action_num"]
            key = f"{d}:{an}"
            if key not in done:
                t = readers[d].get(index[d][an])
                m = masks[p["game"]]
                diff = t["frame"] != t["next_frame"]
                want = (p[side]["n_changed"], p[side]["n_changed_nonticker"])
                got = (int(diff.sum()), int((diff & ~m).sum()))
                if got != want:
                    raise SystemExit(f"{p['pair_id']}/{side}: transition at {key} has "
                                     f"{got} changed cells, pair record says {want}")
                name = hashlib.blake2b(key.encode(), digest_size=8).hexdigest() + ".png"
                render_move(t["frame"], t["next_frame"], m).save(os.path.join(out, name))
                done[key] = name
            mapping[p["pair_id"]][side] = done[key]

    with open(os.path.join(out, "images.json"), "w") as f:
        json.dump(mapping, f)
    print(f"[probe_images] {len(done)} unique move images for {len(pairs)} pairs, "
          f"all verified against the pair records -> {out}/ ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
