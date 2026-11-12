"""tools/probe_engine.py on a made-up engine: one button that moves a dot through an
in-between frame, and an ACTION7 that undoes the previous move."""
import enum
import random
from types import SimpleNamespace

import numpy as np

import probe_engine as P

Act = enum.Enum("Act", {"ACTION1": 1, "ACTION7": 7})


class FakeEnv:
    action_space = [Act.ACTION1, Act.ACTION7]

    def _obs(self, frames):
        return SimpleNamespace(frame=frames, state=SimpleNamespace(name="NOT_FINISHED"),
                               available_actions=[1, 7], levels_completed=0)

    def _board(self, x):
        b = np.zeros((64, 64), np.uint8)
        b[30, x] = 5
        return b

    def reset(self):
        self.x, self.history = 10, []
        return self._obs([self._board(self.x)])

    def step(self, action, data=None):
        if action is Act.ACTION1:                       # the dot moves two cells, through the cell between
            self.history.append(self.x)
            self.x = (self.x + 2) % 60
            return self._obs([self._board(self.x - 1), self._board(self.x)])
        if self.history:                                # ACTION7: back to where the dot was
            self.x = self.history.pop()
        return self._obs([self._board(self.x)])


def test_probe_counts_in_between_frames_and_undos():
    c = P.probe(FakeEnv(), 300, random.Random(0))
    assert c["moves"] == 300 and c["offered"] == {1, 7}
    assert c["multi"] > 0 and c["max_frames"] == 2
    assert c["changing_between"] == c["multi"]          # every 2-frame move passed through another screen
    assert c["still_flash"] == 0
    assert c["a7_after_change"] > 0 and c["a7_undid"] == c["a7_after_change"]   # every undo was exact


def test_probe_without_action7_never_presses_it():
    c = P.probe(FakeEnv(), 100, random.Random(0), allow7=False)
    assert c["a7"] == 0 and c["multi"] == 100
