"""tools/frozen_moves.py: frozen stretches on a made-up recording."""
import numpy as np

import frozen_moves as F


def board(value=0):
    b = np.zeros((64, 64), np.uint8)
    b[10, 10] = value
    return b


def feed(moves):
    """moves: [(changed?, action_num, level)] -> a Stretches fed those moves."""
    s, colour = F.Stretches(), 1
    for changed, num, level in moves:
        before = board(colour)
        colour = colour % 9 + 1 if changed else colour
        s.move(before, board(colour), num, level)
    return s


def test_a_stretch_ends_at_a_change_a_reset_and_a_level_up():
    s = feed([(False, 1, 0), (False, 2, 0), (True, 3, 0),        # 2 frozen, then a change
              (False, 4, 0), (False, 5, 0), (False, 6, 0),       # 3 frozen ...
              (False, 8, 0),                                     # ... a reset between (7 is missing): a new stretch
              (False, 9, 1)])                                    # a level-up: a new stretch again
    assert s.done == [2, 3, 1] and s.run == 1


def test_the_cut_is_what_lies_past_the_first_moves_of_each_stretch():
    s = feed([(False, n, 0) for n in range(1, 11)] + [(True, 11, 0)] + [(False, n, 0) for n in range(12, 16)])
    got = s.summary(after=3)
    assert got == dict(moves=15, longest=10, tail=4, cut=(10 - 3) + (4 - 3), stretches_over=2)


def test_a_decoration_that_ticks_every_move_does_not_hide_a_frozen_screen():
    """A cell that changes on every move is masked after the warm-up (the agent's own
    mask), so moves that only tick it count as frozen."""
    s = F.Stretches()
    a, b = board(1), board(1)
    for n in range(1, 401):
        a, b = b.copy(), b.copy()
        b[0, 0] = n % 2 + 1                       # the ticker
        s.move(a, b, n, 0)
    assert s.run > 150                            # frozen from the warm-up (200 moves) on
