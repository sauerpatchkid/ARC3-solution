"""gridtools.py — small helpers for 64x64 colour-index screens, written once.

These used to be copied between modules (the bar detector in upgrades.py, the
Rulebook's evidence index in wm/evidence.py, the Coach's screen summary), each
copy with its own bugs to find. Everything here is pure numpy, has no state and
knows nothing about agents, flags or the LLM, so any module may import it.

    border_band(width)              the outer band of the screen, where bars live
    tick_cells(diff)                changed cells in a changed blob of <= 2 cells
    connected_components(mask)      8-connected blobs of a bool mask (cell lists)
    label_components(grid, valid)   4-connected single-colour regions (label image)
    screen_objects(raw, masked)     regions worth clicking, most salient first

NOT here, on purpose: metrics_common.find_indicator_cells. That is the published
scorer's decoration detector; it works on whole moves rather than blobs, and
changing it moves every reported baseline number (CLAUDE.md, "Known measurement
caveats"). Two notions of "connected" are kept because their callers differ:
a changed blob is 8-connected (a diagonal tick is one tick), a clickable object
is a 4-connected patch of one colour.
"""
import numpy as np

GRID = 64
BAR_BORDER = 4              # bar cells lie within this many cells of a screen edge
MAX_OBJECTS = 64
MAX_OBJECT_CELLS = 1024     # bigger single-colour regions are background, not objects

_SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1))


def border_band(width=BAR_BORDER):
    """Bool (64, 64): True within `width` cells of any screen edge."""
    band = np.zeros((GRID, GRID), dtype=bool)
    band[:width, :] = band[-width:, :] = True
    band[:, :width] = band[:, -width:] = True
    return band


def _neighbour_sum(m):
    """For each cell, how many of its 8 neighbours are True (numpy, no scipy)."""
    p = np.pad(m.astype(np.int16), 1)
    return sum(p[1 + dy:1 + dy + GRID, 1 + dx:1 + dx + GRID] for dy, dx in _SHIFTS)


def tick_cells(diff):
    """Changed cells that belong to an 8-connected changed component of at most
    2 cells. The shared detector only sees a tick when the WHOLE move changed
    <= 2 cells; this sees a small tick inside a big move (legacy/llm_track/
    tickers.py's idea, vectorised). Size 1: no changed neighbour. Size 2: one
    changed neighbour whose only changed neighbour is this cell."""
    nb = _neighbour_sum(diff)
    single = diff & (nb == 0)
    one = diff & (nb == 1)
    pair = one & (_neighbour_sum(one) == 1)
    return single | pair


def connected_components(mask):
    """8-connected components of a bool mask, as a list of (n, 2) int16 arrays
    of (row, col)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    cells = list(zip(ys.tolist(), xs.tolist()))
    index = {c: i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (y, x), i in index.items():
        for dy, dx in _NEIGHBOURS:
            j = index.get((y + dy, x + dx))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for c, i in index.items():
        groups.setdefault(find(i), []).append(c)
    return [np.array(v, dtype=np.int16) for v in groups.values()]


def label_components(grid, valid):
    """4-connected single-colour components of `grid` among `valid` cells.
    Min-label propagation with pointer jumping; returns int32 labels (-1 where
    not valid), each label the smallest flat index in its component."""
    lab = np.where(valid, np.arange(GRID * GRID, dtype=np.int32).reshape(GRID, GRID), -1)
    big = np.int32(GRID * GRID)
    right = valid[:, :-1] & valid[:, 1:] & (grid[:, :-1] == grid[:, 1:])
    down = valid[:-1, :] & valid[1:, :] & (grid[:-1, :] == grid[1:, :])
    work = np.where(valid, lab, big)
    for _ in range(GRID * 2):
        new = work.copy()
        new[:, :-1] = np.where(right, np.minimum(new[:, :-1], work[:, 1:]), new[:, :-1])
        new[:, 1:] = np.where(right, np.minimum(new[:, 1:], work[:, :-1]), new[:, 1:])
        new[:-1, :] = np.where(down, np.minimum(new[:-1, :], work[1:, :]), new[:-1, :])
        new[1:, :] = np.where(down, np.minimum(new[1:, :], work[:-1, :]), new[1:, :])
        flat = new.ravel()
        v = flat < big
        flat[v] = np.minimum(flat[v], flat[flat[v]])       # pointer jumping
        new = flat.reshape(GRID, GRID)
        if np.array_equal(new, work):
            break
        work = new
    return np.where(valid, work, -1).astype(np.int32)


def screen_objects(raw, masked):
    """Objects worth clicking on this screen, most salient first, as
    (labels, [(label, target_flat_index), ...]). Background (the most common
    colour), masked decoration cells and regions over MAX_OBJECT_CELLS are
    excluded. Salience follows the 3rd-place ARC-AGI-3 agent's idea (size and
    colour rarity): small objects in rare colours first. The target is the
    object's cell nearest its centroid."""
    colours, counts = np.unique(raw[~masked], return_counts=True)
    if len(colours) == 0:
        return np.full((GRID, GRID), -1, np.int32), []
    background = colours[np.argmax(counts)]
    rarity = np.ones(16)
    rarity[colours] = counts / counts.sum()
    valid = (~masked) & (raw != background)
    labels = label_components(raw, valid)
    ys, xs = np.nonzero(labels >= 0)
    if len(ys) == 0:
        return labels, []
    ids, inv, sizes = np.unique(labels[ys, xs], return_inverse=True, return_counts=True)
    cy = np.bincount(inv, weights=ys) / sizes
    cx = np.bincount(inv, weights=xs) / sizes
    d = (ys - cy[inv]) ** 2 + (xs - cx[inv]) ** 2
    order = np.lexsort((d, inv))                      # by object, then distance
    _, first = np.unique(inv[order], return_index=True)
    near = order[first]                               # each object's most central cell
    colour = raw[ys[near], xs[near]].astype(np.int64)
    sal = sizes * rarity[colour]
    keep = sizes <= MAX_OBJECT_CELLS
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(sal[idx], kind="stable")][:MAX_OBJECTS]
    return labels, [(int(ids[i]), int(ys[near[i]]) * GRID + int(xs[near[i]])) for i in idx]
