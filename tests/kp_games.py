"""kp_games.py - Toggle, one made-up game for the Knowledge Planner's tests.

A tiny engine that answers the way the real one does: every move returns a
stack of frames, and the move that completes a level returns the solved board
and then the next level's first screen. It speaks the engine's own enums, so
both the bridge and the reference's own run_episode can play it.

The board: eight one-colour tiles on a lattice around one 3x3 pattern drawn at
2x. A click on a tile turns it from colour A to colour B and back. Each cell of
the pattern is a symbol for the tile on its side of the pattern: SAME (the tile
must have the centre cell's colour) or DIFF (it must not). The level ends when
all eight tiles agree with the pattern. An attempt has a move budget; out of
moves is a game over, and a reset starts the level again.

That is ft09's kind of rule, small enough to read: the reference can learn
"each pattern describes its neighbourhood" from level 1 and plan level 2.
"""
import numpy as np
from arcengine import GameAction, GameState

GRID = 64
BG, A, B = 5, 3, 4          # the background and the two colours a tile takes
SAME, DIFF = 8, 9           # the pattern's two symbols
RING = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

# One level: the centre cell's colour, the 3x3 pattern (its centre cell is not read), and the tiles'
# colours at the start, in RING order. Level 1 starts with every tile wrong, so eight clicks solve it
# and every tile is seen to change.
LEVELS = [
    (A, [[SAME, SAME, DIFF], [SAME, None, DIFF], [DIFF, DIFF, SAME]], [B, B, A, B, A, A, A, B]),
    (B, [[DIFF, SAME, SAME], [DIFF, None, SAME], [SAME, DIFF, DIFF]], [A, A, A, A, A, A, A, A]),
    (A, [[DIFF, DIFF, SAME], [SAME, None, SAME], [DIFF, SAME, DIFF]], [B, A, B, A, B, A, B, A]),
]


def click(y, x):
    """The agent's index for a click at row y, column x."""
    return 5 + GRID * y + x


def tile(dy, dx):
    """The agent's index for a click on the tile at lattice step (dy, dx) from the pattern."""
    return click(29 + 10 * dy, 29 + 10 * dx)


def wanted(k):
    """Level k's solved tiles, in RING order."""
    centre, pattern, _ = LEVELS[k]
    other = B if centre == A else A
    return [centre if pattern[dy + 1][dx + 1] == SAME else other for dy, dx in RING]


class Obs:
    def __init__(self, frames, levels, state, actions):
        self.frame, self.levels_completed, self.state, self.available_actions = frames, levels, state, actions


class Toggle:
    def __init__(self, budget=12, clicks=True):
        self.budget, self.clicks = budget, clicks
        self.action_space = [GameAction.ACTION6] if clicks else [GameAction.ACTION1]
        self.level, self.moves, self.sent = 0, 0, []
        self.tiles = list(LEVELS[0][2])

    def draw(self):
        centre, pattern, _ = LEVELS[self.level]
        g = np.full((GRID, GRID), BG, np.uint8)
        for r in range(3):
            for c in range(3):
                g[27 + 2 * r: 29 + 2 * r, 27 + 2 * c: 29 + 2 * c] = centre if (r, c) == (1, 1) else pattern[r][c]
        for (dy, dx), colour in zip(RING, self.tiles):
            g[28 + 10 * dy: 32 + 10 * dy, 28 + 10 * dx: 32 + 10 * dx] = colour
        return g

    def obs(self, frames, state=GameState.NOT_FINISHED):
        return Obs(frames, self.level, state, list(self.action_space))

    def reset(self):
        """Start the level again (the engine's RESET after a game over)."""
        self.tiles, self.moves = list(LEVELS[self.level][2]), 0
        return self.obs([self.draw()])

    def step(self, action, data=None, reasoning=None):
        """One move, the engine's way. `reasoning` is kept, so a test can read who chose the move."""
        self.sent.append((action.name, dict(data or {}), reasoning))
        self.moves += 1
        if action == GameAction.ACTION6:
            for i, (dy, dx) in enumerate(RING):
                if 28 + 10 * dy <= data["y"] < 32 + 10 * dy and 28 + 10 * dx <= data["x"] < 32 + 10 * dx:
                    self.tiles[i] = B if self.tiles[i] == A else A
        if self.clicks and self.tiles == wanted(self.level):
            solved = self.draw()
            if self.level == len(LEVELS) - 1:
                self.level += 1
                return Obs([solved], self.level, GameState.WIN, list(self.action_space))
            self.level, self.moves = self.level + 1, 0
            self.tiles = list(LEVELS[self.level][2])
            return self.obs([solved, self.draw()])
        if self.moves >= self.budget:
            return self.obs([self.draw()], GameState.GAME_OVER)
        return self.obs([self.draw()])

    def play(self, index):
        """One move given as the agent's index: 0-4 a key, 5 + 64*y + x a click."""
        if index < 5:
            return self.step(getattr(GameAction, f"ACTION{index + 1}"), {})
        y, x = divmod(index - 5, GRID)
        return self.step(GameAction.ACTION6, {"x": x, "y": y})


class Arcade:
    """What the reference's run_episode asks its arcade for."""

    def __init__(self, **kw):
        self.kw, self.env = kw, None

    def make(self, game_id, seed=0, scorecard_id=None, save_recording=False):
        self.env = Toggle(**self.kw)
        return self.env


# Level 1 as a stand-in for Goose: a whole attempt of dead clicks (a game over while the reference only
# watches), then the eight clicks that solve it.
DEAD = click(2, 2)
LEVEL_1 = [DEAD] * 12 + [tile(dy, dx) for dy, dx in RING]
