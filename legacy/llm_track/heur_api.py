"""heur_api.py — what a Qwen-written heuristic may use.

A heuristic is one small function (Phase 1 of the heuristics plan):

    IDEA = "one line saying what it prefers and why"
    def score(board, api):
        buttons, clicks = api.zeros()      # buttons: 5 values (ACTION1-ACTION5)
        ...                                # clicks: 64x64 values, clicks[row, col]
        return buttons, clicks

Higher means more promising, and only the ORDER within one board matters: the
agent standardises the scores per move, and the referee ranks them. `board` is
the current 64x64 grid of colour indices; `np` is numpy; nothing can be imported.

The API is deliberately small and CHEAP. Anything expensive is computed once per
level (`layout`) or once per move by the caller (`history`), so a heuristic that
sticks to numpy on `board` plus these fields runs well under a millisecond, which
is what keeps Goose at full speed. The same class serves the offline referee and
the live agent, so a heuristic behaves identically in grading and in play.

API_DOC below is the exact text Qwen is shown; it lives next to the code so the
two cannot drift apart.
"""
import numpy as np

from .tickers import connected_components

N_BUTTONS = 5
GRID = 64
HISTORY = 8          # earlier moves exposed to a heuristic


class Region:
    """A connected same-colour area of a board (8-connected)."""
    __slots__ = ("colour", "size", "y0", "x0", "y1", "x1", "cells", "mask")

    def __init__(self, colour, cells):
        self.colour = int(colour)
        self.cells = cells                            # (n, 2) int16: (row, col)
        self.mask = np.zeros((GRID, GRID), dtype=bool)   # the region as a board-shaped mask
        self.mask[cells[:, 0], cells[:, 1]] = True
        self.size = int(len(cells))
        self.y0, self.x0 = (int(v) for v in cells.min(axis=0))
        self.y1, self.x1 = (int(v) for v in cells.max(axis=0))

    def __repr__(self):
        return (f"Region(colour={self.colour}, size={self.size}, "
                f"rows {self.y0}-{self.y1}, cols {self.x0}-{self.x1})")


class Move:
    """One earlier move in the current level."""
    __slots__ = ("action", "click", "changed", "move")

    def __init__(self, action, click, changed, move):
        self.action = action        # 1-5 button, 6 click
        self.click = click          # (row, col) or None
        self.changed = changed      # cells that changed, ticker excluded
        self.move = move            # (colour, drow, dcol) of the largest shape that moved, or None

    def __repr__(self):
        return (f"Move(action={self.action}, click={self.click}, "
                f"changed={self.changed}, move={self.move})")


def background_of(board):
    return int(np.bincount(np.asarray(board).ravel(), minlength=16).argmax())


def regions(board, background, ticker):
    """Connected same-colour regions, background colour and ticker excluded,
    largest first. Milliseconds on a busy board: meant for once per level."""
    board = np.asarray(board)
    out = []
    for c in np.unique(board):
        if int(c) == background:
            continue
        m = (board == c) & ~ticker
        if m.any():
            out += [Region(c, cells) for cells in connected_components(m)]
    out.sort(key=lambda r: -r.size)
    return out


class HeuristicAPI:
    """The `api` argument of score(board, api)."""
    __slots__ = ("first", "background", "ticker", "layout", "history", "level",
                 "moves_in_level")

    def __init__(self, first, ticker, layout=None, history=(), level=0, moves_in_level=0):
        self.first = first
        self.ticker = ticker
        self.background = background_of(first)
        self.layout = layout if layout is not None else regions(first, self.background, ticker)
        self.history = list(history)
        self.level = level
        self.moves_in_level = moves_in_level

    @staticmethod
    def zeros():
        return np.zeros(N_BUTTONS), np.zeros((GRID, GRID))


# API_DOC changes are tested on level-1 data only. 2026-09-11 smoke run: 3 of 4
# candidates wrote to buttons[5] or buttons[6] (assuming a button slot for
# ACTION6) and 1 used r.cells as if it were a board-shaped mask - hence the
# explicit ACTION6 note, r.mask, and one example in a different game's style.
API_DOC = """You write ONE Python function. `np` (numpy) is available; nothing can be imported.

    IDEA = "one line saying what the heuristic prefers and why"
    def score(board, api):
        buttons, clicks = api.zeros()   # buttons: 5 values for ACTION1-ACTION5
                                        # clicks: 64x64 values, clicks[row, col]
        ...
        return buttons, clicks

buttons has exactly 5 entries: buttons[0] is ACTION1 ... buttons[4] is ACTION5.
There is NO entry for ACTION6: a click at (row, col) IS ACTION6, and its score
is clicks[row, col].

Higher = more promising. Only the ORDER within one board matters.
`board` is the CURRENT board: a 64x64 numpy array of colour indices 0-15.

`api` gives you:
  api.first           the board at the START of this level (64x64)
  api.background      the most common colour of api.first
  api.ticker          64x64 bool: timer/counter cells that change by themselves
  api.layout          regions on api.first: connected same-colour areas
                      (background and ticker excluded), largest first. Each
                      region r has r.colour, r.size, r.y0, r.x0, r.y1, r.x1
                      (inclusive bounding box), r.mask (a 64x64 bool array
                      that is True on the region) and r.cells (an n x 2 array
                      of (row, col)). board[r.mask] gives the CURRENT colours of
                      that area; clicks[r.mask & (board == 9)] = 1.0 scores the
                      region's cells that are currently colour 9.
  api.history         up to 8 earlier moves in this level, oldest first. Each m
                      has m.action (1-5 button, 6 click), m.click ((row, col) or
                      None), m.changed (cells changed, ticker excluded) and
                      m.move ((colour, drow, dcol) of the largest shape that
                      moved, or None)
  api.level           which level this is (0 = the first)
  api.moves_in_level  how many moves have been made in this level so far

Rules:
- It must average well under 1 millisecond per call: use numpy on whole arrays,
  not Python loops over all 4,096 cells.
- It must not give the same score to every option.
- It is graded on a DIFFERENT level of the same game, so capture the game's rule,
  not one specific board.
- board, api.first, api.ticker and region masks are read-only: use .copy() first
  if you want to modify one.

Example, from a DIFFERENT game - it shows the style (whole-array numpy, no
loops), not the answer for this game:

    IDEA = "prefer clicking cells whose colour is rare on the board"
    def score(board, api):
        buttons, clicks = api.zeros()
        counts = np.bincount(board.ravel(), minlength=16)
        clicks = 1.0 / (1.0 + counts[board])
        clicks[api.ticker] = 0.0
        return buttons, clicks"""
