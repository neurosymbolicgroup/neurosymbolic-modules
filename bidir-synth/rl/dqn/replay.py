"""
Code copied/apdated from
https://github.com/lilianweng/deep-reinforcement-learning-gym/blob/master/playground/policies/memory.py
"""

from typing import List, NamedTuple

import numpy as np

from rl.environment import SynthEnvAction, SynthEnvObservation
from rl.program_search_graph import ProgramSearchGraph


class SynthEnvTransition(NamedTuple):
    obs: SynthEnvObservation
    act: SynthEnvAction
    rew: float
    obs_next: SynthEnvObservation
    done: bool


class SynthEnvReplayBuffer:
    """FIFO circular buffer"""

    # TODO: Improve with ideas from
    # https://danieltakeshi.github.io/2019/07/14/per/

    def __init__(self, max_size: int = 100000, replace: bool = False):
        """
        Buffer contains at most capacity elements.
        Implemented as a circular buffer
        replace controls whether sampling is done with replacement
        """
        self.max_size = max_size
        self.replace = replace

        self._buffer: List[SynthEnvTransition] = []
        self._cur_idx = 0

    @property
    def size(self):
        """How many elements currently in buffer."""
        return len(self._buffer)

    def add(self, t: SynthEnvTransition):
        if self.size < self.max_size:
            self._buffer.append(t)
        else:
            self._buffer[self._cur_idx] = t
        self._cur_idx = (self._cur_idx + 1) % self.max_size

    def sample(self, batch_size: int) -> List[SynthEnvTransition]:
        assert self.size >= batch_size

        idxs = np.random.choice(
            np.arange(self.size),
            size=batch_size,
            replace=self.replace,
        )

        return [self._buffer[i] for i in idxs]
