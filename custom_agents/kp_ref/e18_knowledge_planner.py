"""E18: goal-aware cross-level knowledge transfer on top of StochasticGoose.

Level 1 is played by the unmodified E0 StochasticGoose agent. A passive observer watches every click and what it did.
When level 1 is cleared, three pieces of knowledge are extracted and carried into level 2:

  K1  interactive signatures -- which kinds of objects (colour + scale-normalised shape) changed the state when clicked
  K2  dynamics class        -- are click effects position-based recolourings (a colour-successor table) or moves
                               (tile permutations between fixed slots); chosen by which one explains level 1 consistently
  K3  goal predicate        -- from a small library of relational templates, the instantiation that is TRUE in the
                               level-1 win frame and FALSE at the level-1 start frame

On level 2 the agent probes only objects matching K1, fits the K2 model from those probes, BFS-plans inside the learned
model to a state satisfying K3, executes, and re-plans whenever an observation contradicts a prediction. If no goal
predicate was induced, or planning keeps failing, it falls back to StochasticGoose for the rest of the level.

The predicate *templates* are hand-designed priors (ARC-style relations); which template applies, and how it is grounded
(which colours, which objects play which role, which colour relation), is learned from the level-1 win.
"""
import argparse, hashlib, math, random, time
from collections import defaultdict, deque, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_agi import Arcade
from arcengine import GameAction, GameState

BUDGET = 1500
DEVICE = torch.device("cpu")
NUM_COLOURS = 16
GRID_SIZE = 64
ACTION_NAMES = [f"ACTION{i}" for i in range(1, 6)]
LIVE_GAMES = ["ft09-0d8bbf25", "lp85-305b61c3"]


def get_grid(obs):
    f = np.array(obs.frame)
    if f.ndim == 3:
        f = f[-1]
    return f.astype(int)


def frame_to_onehot(grid):
    t = torch.zeros(NUM_COLOURS, GRID_SIZE, GRID_SIZE, dtype=torch.float32)
    idx = torch.from_numpy(np.clip(grid, 0, NUM_COLOURS - 1).astype(np.int64))
    t.scatter_(0, idx.unsqueeze(0), 1.0)
    return t


# =====================================================================================================================
# E0: StochasticGoose, verbatim from Cross_Level_Knowledge_Transfer.ipynb (Step 6)
# =====================================================================================================================
class CNNBackbone(nn.Module):
    def __init__(self, in_ch=NUM_COLOURS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.out_channels = 48

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return F.relu(self.conv3(h))


class CoordHead(nn.Module):
    def __init__(self, in_channels=48):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels, 24, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, feat):
        h = F.relu(self.conv_a(feat))
        logits = self.conv_b(h)
        logits = F.interpolate(logits, size=(GRID_SIZE, GRID_SIZE), mode="nearest")
        return logits.reshape(logits.size(0), -1)


class ActionHeadCNN(nn.Module):
    def __init__(self, in_channels=48, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, len(ACTION_NAMES))

    def forward(self, feat):
        pooled = feat.mean(dim=(2, 3))
        return self.fc2(F.relu(self.fc1(pooled)))


class StochasticGooseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.action_head = ActionHeadCNN(self.backbone.out_channels)
        self.coord_head = CoordHead(self.backbone.out_channels)

    def forward(self, x):
        feat = self.backbone(x)
        return torch.cat([self.action_head(feat), self.coord_head(feat)], dim=1)


def sample_action(combined_logits, legal, bias=None):
    legal_by_name = {a.name: a for a in legal}
    complex_action = next((a for a in legal if a.is_complex()), None)
    logits = combined_logits.clone()
    if bias is not None:
        logits = logits + bias
    action_logits = logits[:5].clone()
    mask = torch.full((5,), float("-inf"))
    for i, name in enumerate(ACTION_NAMES):
        if name in legal_by_name:
            mask[i] = 0.0
    action_logits = action_logits + mask
    action_probs = torch.sigmoid(action_logits)
    coord_logits = logits[5:]
    coord_probs = torch.sigmoid(coord_logits) / coord_logits.numel() if complex_action is not None else torch.zeros_like(coord_logits)
    all_probs = torch.cat([action_probs, coord_probs])
    total = float(all_probs.sum())
    if not math.isfinite(total) or total <= 0:
        a = random.choice(legal)
        data = {"x": random.randint(0, 63), "y": random.randint(0, 63)} if a.is_complex() else {}
        return a, data, None, False
    probs_np = (all_probs / total).detach().numpy()
    probs_np = probs_np / probs_np.sum()
    idx = int(np.random.choice(len(probs_np), p=probs_np))
    if idx < 5:
        return legal_by_name[ACTION_NAMES[idx]], {}, idx, True
    coord_idx = idx - 5
    y, x = divmod(coord_idx, GRID_SIZE)
    return complex_action, {"x": int(x), "y": int(y)}, idx, True


class StochasticGooseAgent:
    name = "E0_stochasticgoose"

    def __init__(self, seed=0, train_every=4, batch_size=32, buffer_size=2000, lr=1e-3):
        self.seed = seed
        self.train_every = train_every
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr = lr

    def reset(self):
        self.level = 0
        self.buffer = deque(maxlen=self.buffer_size)
        self.seen_hashes = set()
        self.prev_state = None
        self.prev_logit_idx = None
        self.step_count = 0
        self.train_steps = 0
        self.level_clears_log = []
        self.level_actions = 0
        self._new_level_model()

    def _new_level_model(self):
        g = torch.Generator().manual_seed(self.seed * 997 + self.level)
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
        self.model = StochasticGooseModel().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer.clear(); self.seen_hashes.clear()
        self.prev_state = None; self.prev_logit_idx = None

    def note(self, msg):
        pass

    def _check_level_up(self, obs):
        if obs.levels_completed > self.level:
            self.level_clears_log.append((self.level, self.level_actions))
            self.level = obs.levels_completed
            self.level_actions = 0
            self._new_level_model()

    def act(self, obs, legal):
        self._check_level_up(obs)
        grid = get_grid(obs)
        x = frame_to_onehot(grid)
        if self.prev_state is not None:
            changed = not torch.equal(self.prev_state, x)
            self._remember(self.prev_state, self.prev_logit_idx, changed)
        with torch.no_grad():
            combined = self.model(x.unsqueeze(0).to(DEVICE))[0].cpu()
        action, data, logit_idx, ok = sample_action(combined, legal)
        self.prev_state = x.bool()
        self.prev_logit_idx = logit_idx
        self.step_count += 1; self.level_actions += 1
        if self.step_count % self.train_every == 0:
            self._train_step()
        return action, data, f"{self.name}:{action.name}", ok

    def _remember(self, state_bool, logit_idx, changed):
        if logit_idx is None:
            return
        key = hashlib.md5(state_bool.numpy().tobytes() + str(logit_idx).encode()).hexdigest()
        if key in self.seen_hashes:
            return
        self.seen_hashes.add(key)
        self.buffer.append((state_bool, logit_idx, 1.0 if changed else 0.0))

    def _train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states = torch.stack([b[0].float() for b in batch]).to(DEVICE)
        idxs = torch.tensor([b[1] for b in batch], dtype=torch.long, device=DEVICE)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
        self.optimizer.zero_grad()
        combined = self.model(states)
        selected = combined.gather(1, idxs.unsqueeze(1)).squeeze(1)
        F.binary_cross_entropy_with_logits(selected, rewards).backward()
        self.optimizer.step()
        self.train_steps += 1

    def stats(self):
        return dict(train_steps=self.train_steps, level_clears_log=str(self.level_clears_log))


# =====================================================================================================================
# Perception: objects with scale-normalised signatures
# =====================================================================================================================
def _components(mask):
    """4-connected components of a boolean mask -> list of (ys, xs) arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        stack, cells = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys, xs = zip(*cells)
        comps.append((np.array(ys), np.array(xs)))
    return comps


def _run_lengths(line):
    runs, n = [], 1
    for a, b in zip(line[:-1], line[1:]):
        if a == b:
            n += 1
        else:
            runs.append(n); n = 1
    runs.append(n)
    return runs


def normalised_pattern(patch):
    """Downsample a patch by the gcd of all its row/column run lengths, so the same sprite drawn at 1x, 2x, 4x
    yields the same pattern."""
    g = 0
    for row in patch:
        for r in _run_lengths(list(row)):
            g = math.gcd(g, r)
    for col in patch.T:
        for r in _run_lengths(list(col)):
            g = math.gcd(g, r)
    g = max(g, 1)
    return patch[::g, ::g], g


class Obj:
    __slots__ = ("ys", "xs", "pos", "sig", "shape", "colour", "pattern", "scale", "click", "hud", "area")

    def __init__(self, ys, xs, grid):
        self.ys, self.xs = ys, xs
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        self.pos = ((y0 + y1) / 2.0, (x0 + x1) / 2.0)
        patch = np.full((y1 - y0 + 1, x1 - x0 + 1), -1, dtype=int)
        patch[ys - y0, xs - x0] = grid[ys, xs]
        pat, self.scale = normalised_pattern(patch)
        self.pattern = pat
        self.sig = (pat.shape, pat.tobytes())
        self.shape = (pat.shape, (pat >= 0).tobytes())
        cols = set(np.unique(pat[pat >= 0]).tolist())
        self.colour = cols.pop() if len(cols) == 1 else None
        self.area = len(ys)
        i = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
        self.click = (int(xs[i]), int(ys[i]))
        h, w = grid.shape
        thin = (y1 - y0 + 1) == 1 or (x1 - x0 + 1) == 1
        touches = y0 == 0 or x0 == 0 or y1 == h - 1 or x1 == w - 1
        self.hud = thin and touches and max(y1 - y0, x1 - x0) + 1 >= h // 2


def parse_objects(grid, big_frac=0.08):
    """Background = every colour that has a single connected region covering > big_frac of the frame.
    Objects = 4-connected (multi-colour) components of everything else. HUD = long thin lines on the border."""
    h, w = grid.shape
    bg = set()
    for c in np.unique(grid):
        for ys, xs in _components(grid == c):
            if len(ys) > big_frac * h * w:
                bg.add(int(c)); break
    mask = ~np.isin(grid, list(bg))
    objs = [Obj(ys, xs, grid) for ys, xs in _components(mask)]
    return [o for o in objs if not o.hud], bg


def object_at(objs, x, y):
    for o in objs:
        if np.any((o.ys == y) & (o.xs == x)):
            return o
    return None


# =====================================================================================================================
# K2: learned position-based effect model over object slots
# =====================================================================================================================
class SlotState:
    """A frame as an ordered tuple of signature ids over fixed slots (object centre positions)."""

    def __init__(self, slots, sig_ids):
        self.slots = slots          # list of (y, x)
        self.sig_ids = sig_ids      # tuple of int


class EffectModel:
    """Click effects as fixed maps over slots: new[q] = T(old[src[q]]), with T = identity (move class) or the
    colour-successor table (recolour class). Learned per click target; state-independent by assumption (K2)."""

    def __init__(self, dyn_class, succ=None, period=None):
        self.dyn_class = dyn_class                  # "recolour" or "move"
        self.succ = dict(succ or {})                # sig -> sig (recolour class)
        self.period = period                        # transferred colour-cycle length (2 = involution), or None
        self.cands = {}                             # action key -> list[set(src slot idx)] (move class)
        self.recol = {}                             # action key -> frozenset(slot idx)       (recolour class)
        self.noop = set()                           # action keys observed to change nothing
        self._src_cache = {}

    def observe(self, akey, pre, post, sig_of):
        if pre == post:
            self.noop.add(akey); return
        self.noop.discard(akey)
        if self.dyn_class == "recolour":
            changed = frozenset(i for i, (a, b) in enumerate(zip(pre, post)) if a != b)
            for i in changed:
                self.succ[pre[i]] = post[i]
                if self.period == 2:                # level-1 cycles were 2-cycles: a->b implies b->a
                    self.succ.setdefault(post[i], pre[i])
            self.recol[akey] = changed
            return
        n = len(pre)
        self._src_cache.pop(akey, None)
        prev = self.cands.get(akey)
        new = []
        for q in range(n):
            c = {p for p in range(n) if pre[p] == post[q]}
            if prev is not None and prev[q] & c:
                c = prev[q] & c
            new.append(c)
        self.cands[akey] = new

    def known(self, akey):
        return akey in self.noop or akey in self.recol or akey in self.cands

    def src_map(self, akey, slots):
        """Most plausible bijection consistent with every observation so far: min total squared displacement
        ('objects move smoothly'), which prefers a one-step rotation over identity-plus-long-jumps when a run of
        same-coloured tiles hides part of the motion."""
        cached = self._src_cache.get(akey)
        if cached is not None:
            return cached
        cands = self.cands[akey]
        n = len(cands)
        involved = sorted({q for q, c in enumerate(cands) if c != {q}} | {p for q, c in enumerate(cands) if c != {q} for p in c})
        src = list(range(n))
        if involved:
            big = 1e9
            cost = [[big] * len(involved) for _ in involved]
            for i, q in enumerate(involved):
                for j, p in enumerate(involved):
                    if p in cands[q]:
                        d = abs(slots[p][0] - slots[q][0]) + abs(slots[p][1] - slots[q][1])
                        cost[i][j] = d * d
            for i, j in enumerate(hungarian(cost)):
                src[involved[i]] = involved[j]
        self._src_cache[akey] = src
        return src

    def compile(self, slots):
        """Return {akey: fn(state_tuple) -> state_tuple or None}."""
        fns = {}
        for akey, changed in self.recol.items():
            fns[akey] = ("recolour", tuple(sorted(changed)))
        for akey in self.cands:
            fns[akey] = ("move", tuple(self.src_map(akey, slots)))
        return fns

    def apply(self, compiled, state):
        kind, data = compiled
        if kind == "move":
            return tuple(state[p] for p in data)
        s = list(state)
        for i in data:
            nxt = self.succ.get(s[i])
            if nxt is None:
                return None
            s[i] = nxt
        return tuple(s)


def hungarian(cost):
    """Min-cost perfect assignment for a square matrix; returns col index assigned to each row."""
    n = len(cost)
    INF = float("inf")
    u, v, p, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [INF] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    ans = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            ans[p[j] - 1] = j - 1
    return ans


def infer_dynamics_class(transitions):
    """K2: does a single, global colour-successor table explain every observed change (recolour class)?
    Otherwise the effects are moves between slots."""
    succ = {}
    for pre, post in transitions:
        if len(pre) != len(post):
            return "move", {}
        for a, b in zip(pre, post):
            if a != b:
                if succ.get(a, b) != b:
                    return "move", {}
                succ[a] = b
    return "recolour", succ


def cycle_period(succ):
    """Common length of the colour cycles in a successor table (None if open chains or mixed lengths)."""
    lengths = set()
    for a in succ:
        n, b = 1, succ[a]
        while b != a and b in succ and n <= len(succ):
            b = succ[b]; n += 1
        if b != a:
            return None
        lengths.add(n)
    return lengths.pop() if len(lengths) == 1 else None


# =====================================================================================================================
# K3: relational goal templates, grounded and selected from the level-1 win
# =====================================================================================================================
class Grounded:
    """A goal predicate grounded on a concrete layout: test(state_tuple) -> bool, heur(state_tuple) -> float."""

    def __init__(self, name, test, heur, detail, relevant_colours=None):
        self.name, self.test, self.heur, self.detail = name, test, heur, detail
        self.relevant_colours = relevant_colours     # predicate reads only slots holding these colours (or None)


def _pitch(slots):
    if len(slots) < 2:
        return 1.0
    a = np.array(slots, dtype=float)
    d = np.abs(a[:, None, :] - a[None, :, :]).sum(-1)
    np.fill_diagonal(d, np.inf)
    nn_ = d.min(1)
    return float(np.median(nn_[np.isfinite(nn_)]))


class EnclosureTemplate:
    """'Every frame of 4 same-coloured static corner marks must enclose a movable object of that same colour.'"""
    name = "colour_matched_enclosure"

    def ground(self, layout, colour_relation="same"):
        groups = []
        by_colour = defaultdict(list)
        for i in layout["static"]:
            o = layout["objs"][i]
            if o.colour is not None and o.pattern.size == 1:
                by_colour[(o.colour, o.area)].append(o.pos)
        for (col, _), pts in by_colour.items():
            pts = set(pts)
            for (y0, x0) in pts:
                for (y1, x1) in pts:
                    if y1 > y0 and x1 > x0 and (y0, x1) in pts and (y1, x0) in pts:
                        corners = {(y0, x0), (y0, x1), (y1, x0), (y1, x1)}
                        if any(y0 <= py <= y1 and x0 <= px <= x1 for py, px in pts - corners):
                            continue        # not minimal: another mark lies inside / on an edge
                        inner = [p for p, (sy, sx) in enumerate(layout["slots"]) if y0 < sy < y1 and x0 < sx < x1]
                        groups.append((col, (y0, x0, y1, x1), inner))
        # every mark belongs to one frame: take rectangles smallest-first, each corner used at most once
        used, chosen = set(), []
        for col, (y0, x0, y1, x1), inner in sorted(groups, key=lambda g: (g[1][2] - g[1][0]) * (g[1][3] - g[1][1])):
            corners = {(col, y0, x0), (col, y0, x1), (col, y1, x0), (col, y1, x1)}
            if corners & used:
                continue
            used |= corners
            chosen.append((col, (y0, x0, y1, x1), inner))
        groups = [g for g in chosen if g[2]]
        if not groups:
            return None
        colour_of = layout["colour_of"]
        slots = layout["slots"]
        dyn = layout["dynamic"]

        def test(state):
            for col, _, inner in groups:
                ok = any(colour_of[state[p]] == col for p in inner) if colour_relation == "same" else \
                    any(colour_of[state[p]] is not None for p in inner)
                if not ok:
                    return False
            return True

        pitch = _pitch([slots[i] for i in dyn]) or 1.0

        def heur(state):
            h = 0.0
            for col, (y0, x0, y1, x1), inner in groups:
                cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
                best = min((abs(slots[p][0] - cy) + abs(slots[p][1] - cx) for p in dyn if colour_of[state[p]] == col),
                           default=99.0)
                h += best / pitch
            return h

        return Grounded(self.name, test, heur, f"{len(groups)} frame(s), colours {[g[0] for g in groups]}",
                        relevant_colours={g[0] for g in groups} if colour_relation == "same" else None)


class LegendTemplate:
    """'A static k x k multi-colour key describes its neighbourhood on the tile lattice: each key pixel colour
    stands for a colour relation (same as / different from the key centre, or an absolute colour) that the tile
    at that lattice offset must satisfy.'  The pixel->relation mapping is learned from the win frame."""
    name = "key_legend"

    def __init__(self, mapping=None):
        self.mapping = mapping      # key-pixel colour -> ("same",) | ("diff",) | ("abs", colour)

    def _keys(self, layout):
        objs, slots = layout["objs"], layout["slots"]
        dyn = layout["dynamic"]
        pitch = _pitch([slots[i] for i in dyn])
        keys = []
        for i in layout["static"]:
            o = objs[i]
            k = o.pattern.shape[0]
            if o.pattern.shape != (k, k) or k < 3 or k % 2 == 0 or o.colour is not None or (o.pattern < 0).any():
                continue
            c = k // 2
            cells = []
            for r in range(k):
                for cc in range(k):
                    if (r, cc) == (c, c):
                        continue
                    ty, tx = o.pos[0] + (r - c) * pitch, o.pos[1] + (cc - c) * pitch
                    hit = [p for p in dyn if abs(slots[p][0] - ty) + abs(slots[p][1] - tx) <= pitch / 3]
                    if hit:
                        cells.append((int(o.pattern[r, cc]), hit[0]))
            if cells:
                keys.append((int(o.pattern[c, c]), cells))
        return keys

    def learn_mapping(self, layout, state):
        """Pick, per key-pixel colour, the relation that holds at EVERY occurrence in the win state (relative first)."""
        keys = self._keys(layout)
        if not keys:
            return None
        colour_of = layout["colour_of"]
        obs = defaultdict(list)
        for centre, cells in keys:
            for v, p in cells:
                obs[v].append((centre, colour_of[state[p]]))
        mapping = {}
        for v, pairs in obs.items():
            if all(t == c for c, t in pairs):
                mapping[v] = ("same",)
            elif all(t != c for c, t in pairs):
                mapping[v] = ("diff",)
            elif len({t for _, t in pairs}) == 1:
                mapping[v] = ("abs", pairs[0][1])
            else:
                return None
        return mapping

    def ground(self, layout, mapping=None):
        mapping = mapping or self.mapping
        keys = self._keys(layout)
        if not keys or not mapping:
            return None
        colour_of = layout["colour_of"]
        conds = []
        for centre, cells in keys:
            for v, p in cells:
                rel = mapping.get(v)
                if rel is None:
                    return None
                conds.append((p, rel, centre))

        def ok(state, p, rel, centre):
            return legend_ok(colour_of[state[p]], rel, centre)

        def test(state):
            return all(ok(state, p, rel, c) for p, rel, c in conds)

        def heur(state):
            return float(sum(not ok(state, p, rel, c) for p, rel, c in conds))

        g = Grounded(self.name, test, heur, f"{len(keys)} key(s), mapping {mapping}")
        g.conds = conds
        return g


def legend_ok(colour, rel, centre):
    return colour == centre if rel[0] == "same" else (colour != centre if rel[0] == "diff" else colour == rel[1])


class UniformTemplate:
    """'All movable/recolourable tiles share one colour.' (A common lights-out goal; included as a competing
    hypothesis so that template selection is a real choice.)"""
    name = "uniform_tiles"

    def ground(self, layout):
        dyn = layout["dynamic"]
        colour_of = layout["colour_of"]
        if len(dyn) < 2:
            return None

        def test(state):
            return len({colour_of[state[p]] for p in dyn}) == 1

        def heur(state):
            cnt = Counter(colour_of[state[p]] for p in dyn)
            return float(len(dyn) - max(cnt.values()))

        return Grounded(self.name, test, heur, f"{len(dyn)} tiles")


# =====================================================================================================================
# Knowledge observer + planner
# =====================================================================================================================
class LevelLayout:
    """Slots/static/dynamic bookkeeping for one level, built from the level's first frame."""

    def __init__(self, grid, sig_table):
        objs, self.bg = parse_objects(grid)
        self.objs = objs
        self.slots = [o.pos for o in objs]
        self.sig_table = sig_table
        self.slot_index = {p: i for i, p in enumerate(self.slots)}
        self.dynamic_set = set()
        self.parts = []                 # single-colour sub-objects of multi-colour components (touching sprites)
        for o in objs:
            if o.colour is None:
                cells = np.zeros(grid.shape, dtype=bool); cells[o.ys, o.xs] = True
                for c in np.unique(grid[o.ys, o.xs]):
                    for ys, xs in _components(cells & (grid == c)):
                        self.parts.append(Obj(ys, xs, grid))

    def clickable(self, pos):
        return next((o for o in self.objs + self.parts if o.pos == pos), None)

    def encode(self, grid):
        """Frame -> state tuple over this level's slots (None if the slot structure no longer matches)."""
        objs, _ = parse_objects(grid)
        by_pos = {o.pos: o for o in objs}
        if set(by_pos) != set(self.slots):
            return None, objs
        return tuple(self.sig_table.intern(by_pos[p]) for p in self.slots), objs

    def as_dict(self, model_dyn=None):
        dyn = sorted(self.dynamic_set if model_dyn is None else model_dyn)
        static = [i for i in range(len(self.slots)) if i not in set(dyn)]
        return dict(objs=self.objs, slots=self.slots, dynamic=dyn, static=static,
                    colour_of=self.sig_table.colour_of)


class SigTable:
    def __init__(self):
        self.ids, self.colour_of, self.shape_of, self.objs = {}, {-1: None}, {}, {}

    def intern(self, o):
        if o.sig not in self.ids:
            i = len(self.ids)
            self.ids[o.sig] = i
            self.colour_of[i] = o.colour
            self.shape_of[i] = o.shape
        return self.ids[o.sig]


class KnowledgePlannerAgent(StochasticGooseAgent):
    """E18. Level 1: E0 untouched + passive observer. Level >= 2: K1-guided probing, K2 model, K3-goal BFS planning,
    with StochasticGoose as the fallback. `use_goal=False` gives the K1+K2-only ablation (model-based novelty search)."""
    name = "E18_knowledge_planner"

    def __init__(self, seed=0, use_goal=True, max_plan_failures=30, max_mispredictions=40, bfs_cap=300_000, **kw):
        super().__init__(seed=seed, **kw)
        self.max_mispredictions = max_mispredictions
        self.use_goal = use_goal
        self.max_plan_failures = max_plan_failures
        self.bfs_cap = bfs_cap
        if not use_goal:
            self.name = "E18b_k1k2_only"

    # -- lifecycle ---------------------------------------------------------------------------------------------------
    def reset(self):
        super().reset()
        self.sigs = SigTable()
        self.mode = "goose"
        self.log = []
        # K1: per-shape click outcomes
        self.k1_clicks = Counter(); self.k1_changed = Counter()
        self.k1_shapes, self.k1_sigs = set(), set()
        self.k2_class, self.k2_succ, self.k2_period = None, {}, None
        self.k3 = []                 # list of (template, learned params)
        self._begin_level(None)
        self.knowledge_log = []

    def _begin_level(self, grid):
        self.layout = None
        self.level_start_grid = grid
        self.level_transitions = []  # (akey, pre_state, post_state)
        self.pending = None          # (akey, pre_state, predicted)
        self.effects = None
        self.plan = []
        self.probe_queue = []
        self.goal = None
        self.plan_failures = 0
        self.mispredictions = 0
        self.visited = set()
        self.goal_tried = []

    def note(self, msg):
        if "GAME_OVER" in msg:
            self.plan = []
            self.pending = None

    # -- observation hooks ---------------------------------------------------------------------------------------------
    def _observe_click(self, grid):
        """Called with the grid that resulted from the previous click."""
        if self.pending is None or self.layout is None:
            return
        akey, pre, predicted = self.pending
        self.pending = None
        post, _ = self.layout.encode(grid)
        if pre is None or post is None:
            return
        self.level_transitions.append((akey, pre, post))
        if pre != post:
            for i, (a, b) in enumerate(zip(pre, post)):
                if a != b:
                    self.layout.dynamic_set.add(i)
        if self.effects is not None:
            self.effects.observe(akey, pre, post, self.sigs)
            if predicted is not None and predicted != post:
                self.plan = []
                self.mispredictions += 1
                self.log.append(("mispredict", akey))
        self.visited.add(post)

    def _record_click(self, grid, data, predicted=None):
        if self.layout is None or not data:
            self.pending = None
            return None
        objs = self.layout_objs_cache
        o = object_at(objs, data["x"], data["y"])
        akey = o.pos if o is not None else ("px", data["x"], data["y"])
        pre, _ = self.layout.encode(grid)
        self.pending = (akey, pre, predicted)
        return o

    # -- knowledge extraction at level-up ------------------------------------------------------------------------------
    def _extract_knowledge(self, win_grid):
        lay = self.layout
        if lay is None:
            return
        # K1: which object shapes/sigs changed the state when clicked
        for akey, pre, post in self.level_transitions:
            if isinstance(akey, tuple) and akey and akey[0] == "px":
                continue
            i = lay.slot_index.get(akey)
            if i is None:
                continue
            o = lay.objs[i]
            self.k1_clicks[o.shape] += 1
            if pre != post:
                self.k1_changed[o.shape] += 1
                self.k1_sigs.add(o.sig)
        self.k1_shapes |= {s for s in self.k1_clicks if self.k1_changed[s] > 0}
        # K2: dynamics class over the transitions that changed something
        changes = [(pre, post) for _, pre, post in self.level_transitions if pre != post]
        if changes:
            self.k2_class, succ = infer_dynamics_class(changes)
            self.k2_succ.update(succ)
            self.k2_period = cycle_period(succ) if self.k2_class == "recolour" else None
        # K3: templates true at the win frame, false at the level start
        win_state, _ = lay.encode(win_grid) if win_grid is not None else (None, None)
        start_state, _ = lay.encode(self.level_start_grid)
        k3 = []
        if win_state is not None and start_state is not None:
            d = lay.as_dict()
            cands = []
            g = EnclosureTemplate().ground(d)
            if g is not None:
                cands.append((EnclosureTemplate(), {}, g))
            lt = LegendTemplate()
            mapping = lt.learn_mapping(d, win_state)
            if mapping:
                g = lt.ground(d, mapping)
                if g is not None:
                    cands.append((LegendTemplate(mapping), {"mapping": mapping}, g))
            g = UniformTemplate().ground(d)
            if g is not None:
                cands.append((UniformTemplate(), {}, g))
            for tmpl, params, g in cands:
                verdict = (g.test(win_state), g.test(start_state))
                self.knowledge_log.append((self.level, tmpl.name, g.detail, f"win={verdict[0]} start={verdict[1]}"))
                if verdict == (True, False):
                    k3.append(tmpl)
        if k3:
            self.k3 = k3
        self.knowledge_log.append((self.level, "K1", f"{len(self.k1_shapes)} interactive shape(s)", ""))
        self.knowledge_log.append((self.level, "K2", self.k2_class, f"succ={len(self.k2_succ)} period={self.k2_period}"))
        self.knowledge_log.append((self.level, "K3", [t.name for t in self.k3], ""))

    # -- main act --------------------------------------------------------------------------------------------------------
    def act(self, obs, legal):
        grid = get_grid(obs)
        if obs.levels_completed > self.level:
            f = np.array(obs.frame)
            win_grid = f[-2].astype(int) if f.ndim == 3 and len(f) >= 2 else None
            self._observe_click(win_grid if win_grid is not None else grid)
            self._extract_knowledge(win_grid)
            self._check_level_up(obs)   # E0's own level-up bookkeeping (fresh Goose model), same as E0 would do
            have_knowledge = bool(self.k1_shapes) and self.k2_class is not None and (self.k3 or not self.use_goal)
            self.mode = "plan" if have_knowledge else "goose"
            self._begin_level(grid)
        else:
            self._observe_click(grid)

        if self.layout is None or (self.level_start_grid is None):
            if self.level_start_grid is None:
                self.level_start_grid = grid
            self.layout = LevelLayout(grid, self.sigs)
            if self.mode == "plan":
                self._init_planning()

        self.layout_objs_cache = parse_objects(grid)[0]

        if self.mode == "plan":
            out = self._plan_act(grid, legal)
            if out is not None:
                return out
            self.mode = "goose"
            self.log.append(("fallback-to-goose", self.level))

        # E0 path (identical to StochasticGooseAgent.act), plus passive recording
        action, data, raw, ok = super().act(obs, legal)
        self._record_click(grid, data)
        return action, data, raw, ok

    # -- planning ----------------------------------------------------------------------------------------------------------
    def _init_planning(self):
        self.effects = EffectModel(self.k2_class, succ=self.k2_succ, period=self.k2_period)
        objs = self.layout.objs + [o for o in self.layout.parts           # parts only on an exact K1 match
                                   if o.pos not in self.layout.slot_index and o.sig in self.k1_sigs]
        exact = [o for o in objs if o.sig in self.k1_sigs]
        shape = [o for o in objs if o.shape in self.k1_shapes and o.sig not in self.k1_sigs]
        self.probe_queue = [o.pos for o in exact + shape]
        self.log.append(("probe-set", self.level, len(exact), len(shape), len(objs)))

    def _click(self, legal, pos, tag, grid, predicted=None):
        complex_action = next((a for a in legal if a.is_complex()), None)
        o = self.layout.clickable(pos)
        data = {"x": o.click[0], "y": o.click[1]}
        pre, _ = self.layout.encode(grid)
        self.pending = (pos, pre, predicted)
        self.step_count += 1; self.level_actions += 1
        return complex_action, data, f"{self.name}:{tag}({data['x']},{data['y']})", True

    def _ground_goal(self):
        d = self.layout.as_dict(model_dyn=self._model_dynamic())
        for tmpl in self.k3:
            if tmpl in self.goal_tried:
                continue
            g = tmpl.ground(d)
            if g is not None:
                return tmpl, g
        return None, None

    def _model_dynamic(self):
        dyn = set(self.layout.dynamic_set)
        for akey, ch in self.effects.recol.items():
            dyn |= set(ch)
        for akey in self.effects.cands:
            src = self.effects.src_map(akey, self.layout.slots)
            dyn |= {q for q, p in enumerate(src) if p != q}
        return dyn

    def _plan_act(self, grid, legal):
        state, _ = self.layout.encode(grid)
        if state is None:
            return None
        complex_action = next((a for a in legal if a.is_complex()), None)
        if complex_action is None:
            return None
        self.visited.add(state)

        # 1) probe K1-matching objects whose effect is still unknown
        while self.probe_queue:
            pos = self.probe_queue.pop(0)
            if not self.effects.known(pos):
                return self._click(legal, pos, "probe", grid)

        if self.plan_failures > self.max_plan_failures or self.mispredictions > self.max_mispredictions:
            return None
        compiled = self.effects.compile(self.layout.slots)
        compiled = {k: v for k, v in compiled.items() if k not in self.effects.noop}
        if not compiled:
            return None

        # 2) follow / make a plan
        if not self.plan:
            if self.use_goal:
                if self.goal is None:
                    tmpl, self.goal = self._ground_goal()
                    if self.goal is None:
                        return None
                    self.goal_tmpl = tmpl
                    self.log.append(("goal", self.level, self.goal.name, self.goal.detail))
                plan = self._plan_to_goal(state, compiled)
                if plan is None:
                    # model may still be too wrong to see a route: gather data with a novelty step and retry
                    self.log.append(("no-plan", self.level, self.goal.name))
                    self.plan_failures += 1
                    if self.plan_failures > self.max_plan_failures:
                        self.goal_tried.append(self.goal_tmpl)
                        self.goal = None
                        return None
                    plan = self._bfs(state, compiled, lambda s: s not in self.visited, novelty=True)
                    if plan is None:
                        return None
                    plan = plan[:1]
                else:
                    self.log.append(("plan", self.level, len(plan)))
            else:
                plan = self._bfs(state, compiled, lambda s: s not in self.visited, novelty=True)
                if plan is None:
                    return None
            self.plan = plan
        akey, predicted = self.plan.pop(0)
        return self._click(legal, akey, "plan", grid, predicted=predicted)

    def _solve_gf2(self, state, compiled):
        """Period-2 recolour dynamics (transferred K2) compose as XOR, so per-slot goal conditions become a linear
        system over GF(2): exact for self-toggles and lights-out-style neighbour patterns, at any number of tiles."""
        acts = [a for a, (kind, _) in compiled.items() if kind == "recolour"]
        touch = {a: set(compiled[a][1]) for a in acts}
        colour_of, succ = self.sigs.colour_of, self.effects.succ
        need = {}
        for p, rel, centre in self.goal.conds:
            cur = state[p]
            alt = succ.get(cur)
            ok0 = legend_ok(colour_of[cur], rel, centre)
            ok1 = alt is not None and legend_ok(colour_of[alt], rel, centre)
            if ok0 and ok1:
                continue
            if not ok0 and not ok1:
                return None
            bit = 1 if ok1 else 0
            if need.get(p, bit) != bit:
                return None
            need[p] = bit
        rows = []
        for p, bit in need.items():
            mask = sum(1 << j for j, a in enumerate(acts) if p in touch[a])
            rows.append([mask, bit])
        pivots, r = [], 0
        for j in range(len(acts)):
            piv = next((i for i in range(r, len(rows)) if rows[i][0] >> j & 1), None)
            if piv is None:
                continue
            rows[r], rows[piv] = rows[piv], rows[r]
            for i in range(len(rows)):
                if i != r and rows[i][0] >> j & 1:
                    rows[i][0] ^= rows[r][0]; rows[i][1] ^= rows[r][1]
            pivots.append(j); r += 1
        if any(m == 0 and b for m, b in rows[r:]):
            return None
        x = [0] * len(acts)
        for i, j in enumerate(pivots):
            x[j] = rows[i][1]
        plan, s_ = [], state
        for j, a in enumerate(acts):
            if x[j]:
                s_ = self.effects.apply(compiled[a], s_)
                if s_ is None:
                    return None
                plan.append((a, s_))
        return plan if plan and self.goal.test(s_) else None

    def _plan_to_goal(self, state, compiled):
        if getattr(self.goal, "conds", None) is not None and self.effects.dyn_class == "recolour" \
                and self.effects.period == 2:
            plan = self._solve_gf2(state, compiled)
            if plan is not None:
                return plan
        rel = self.goal.relevant_colours
        if self.effects.dyn_class != "move" or not rel:
            return self._bfs(state, compiled, self.goal.test)
        # exact projection: under slot permutations, only the objects the predicate reads need tracking
        colour_of = self.sigs.colour_of
        n = len(state)
        dests = {}
        for akey, (kind, src) in compiled.items():
            d = [0] * n
            for q, p in enumerate(src):
                d[p] = q
            dests[akey] = d
        start = tuple(sorted((i, s) for i, s in enumerate(state) if colour_of[s] in rel))

        def full(proj):
            f = [-1] * n
            for i, s in proj:
                f[i] = s
            return tuple(f)

        if self.goal.test(full(start)):
            return None
        q = deque([start]); parent = {start: None}
        while q:
            cur = q.popleft()
            for akey, d in dests.items():
                nxt = tuple(sorted((d[i], s) for i, s in cur))
                if nxt in parent:
                    continue
                parent[nxt] = (cur, akey)
                if self.goal.test(full(nxt)):
                    acts = []
                    while parent[nxt] is not None:
                        prev, a = parent[nxt]
                        acts.append(a); nxt = prev
                    acts.reverse()
                    plan, s_ = [], state             # re-expand with the full model so predictions can be checked
                    for a in acts:
                        s_ = self.effects.apply(compiled[a], s_)
                        plan.append((a, s_))
                    return plan
                if len(parent) > self.bfs_cap:
                    return None
                q.append(nxt)
        return None

    def _bfs(self, start, compiled, goal_test, novelty=False):
        if not novelty and goal_test(start):
            return None
        q = deque([start]); parent = {start: None}
        while q:
            s = q.popleft()
            for akey, c in compiled.items():
                n = self.effects.apply(c, s)
                if n is None or n in parent:
                    continue
                parent[n] = (s, akey)
                if goal_test(n):
                    path = []
                    while parent[n] is not None:
                        ps, a = parent[n]
                        path.append((a, n)); n = ps
                    return path[::-1]
                if len(parent) > self.bfs_cap:
                    return None
                q.append(n)
        return None

    def stats(self):
        s = super().stats()
        s["knowledge"] = str(self.knowledge_log)
        s["plan_log"] = str(self.log[:40])
        return s


# =====================================================================================================================
# Harness: same as the notebook's run_episode (Step 5), so results are comparable to E0/E13/E14 CSVs
# =====================================================================================================================
def run_episode(arc, game_id, agent, scorecard_id, max_steps=BUDGET, seed=0):
    game_hash = int(hashlib.md5(game_id.encode()).hexdigest(), 16) % 100_000
    combined_seed = (seed * 100_003 + game_hash) % (2**31 - 1)
    random.seed(combined_seed); np.random.seed(combined_seed); torch.manual_seed(combined_seed)

    env = arc.make(game_id, seed=seed, scorecard_id=scorecard_id, save_recording=False)
    obs = env.reset()
    agent.reset()
    actions_used, parse_fail = 0, 0
    prev = get_grid(obs)
    seen, changed, first_clear, second_clear, third_clear = {prev.tobytes()}, 0, None, None, None
    t0 = time.time()
    while actions_used < max_steps:
        legal = [a for a in env.action_space if a != GameAction.RESET]
        if not legal:
            break
        action, data, raw, ok = agent.act(obs, legal)
        parse_fail += (not ok)
        new_obs = env.step(action, data=data, reasoning={"raw": raw[:300]})
        actions_used += 1
        if new_obs is None:
            break
        obs = new_obs
        g = get_grid(obs)
        changed += int(g.shape != prev.shape or bool((g != prev).any()))
        seen.add(g.tobytes()); prev = g
        if obs.levels_completed >= 1 and first_clear is None:
            first_clear = actions_used
        if obs.levels_completed >= 2 and second_clear is None:
            second_clear = actions_used
        if obs.levels_completed >= 3 and third_clear is None:
            third_clear = actions_used
        if obs.state == GameState.WIN:
            break
        if obs.state == GameState.GAME_OVER:
            obs = env.reset() or obs
            actions_used += 1
            agent.note("GAME_OVER -> level restarted")
            prev = get_grid(obs)
    row = dict(game=game_id, agent=agent.name, levels=obs.levels_completed, actions=actions_used,
               state=obs.state.name, parse_fail=parse_fail, seconds=round(time.time() - t0, 1),
               unique_states=len(seen), changed_frac=round(changed / max(1, actions_used), 3),
               first_clear=first_clear, second_clear=second_clear, third_clear=third_clear,
               level2_actions=(second_clear - first_clear) if second_clear and first_clear else None)
    if hasattr(agent, "stats"):
        row.update(agent.stats())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="E18", choices=["E0", "E18", "E18b"])
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--games", default=",".join(LIVE_GAMES))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    arc = Arcade()
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for g in args.games.split(","):
            if args.variant == "E0":
                agent = StochasticGooseAgent(seed=seed)
            else:
                agent = KnowledgePlannerAgent(seed=seed, use_goal=(args.variant == "E18"))
            sc = arc.create_scorecard(tags=[args.variant, f"s{seed}"])
            r = run_episode(arc, g, agent, sc, max_steps=args.budget, seed=seed)
            r["seed"] = seed
            try:
                arc.close_scorecard(sc)
            except Exception:
                pass
            print(f"{r['agent']:24s} s{seed} {g}: levels={r['levels']} first={r['first_clear']} "
                  f"second={r['second_clear']} third={r['third_clear']} t={r['seconds']}s", flush=True)
            if "plan_log" in r:
                print("   knowledge:", r["knowledge"][:900]); print("   log:", r["plan_log"][:900], flush=True)
            rows.append(r)
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    print(df[["agent", "game", "seed", "levels", "first_clear", "second_clear", "third_clear"]].to_string(index=False))


if __name__ == "__main__":
    main()
