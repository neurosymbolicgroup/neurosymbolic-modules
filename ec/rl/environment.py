from typing import NamedTuple, Optional, Tuple

import gym

from rl.operations import Op
from bidir.primitives.types import Grid
from rl.program_search_graph import ProgramSearchGraph, ValueNode

SynthAction = Tuple[Op, Tuple[Optional[ValueNode], ...]]


class SynthEnvObservation(NamedTuple):
    psg: ProgramSearchGraph
    action_count: int


class SynthEnv(gym.Env):
    """
    Reward:
        Gives reward when task solved.
        Gives penalty when task timeout.
    Starting State:
        The root and leaf of the tree.
    State:
        The developing tree, and actions taken to get there.
    """
    def __init__(
        self,
        train_examples: Tuple[Tuple[Any, Any], ...],
        test_examples: Tuple[Tuple[Any, Any], ...],
        max_actions=100,
        timeout_penalty=-1,
    ):
        """
        Initialize the environment for given set of training and test examples.

        All ops are provided at the beginning as well.

        Maybe we should eventually allow for arbitrary synthesis tasks, not
        just Grid -> Grid tasks.
        """
        self.max_actions = max_actions
        self.action_count = 0
        self.timeout_penalty = -1

        # currently only train examples supported
        in_grids, out_grids = zip(*train_examples)
        self.psg = ProgramSearchGraph(in_grids, out_grids)

    @property
    def done(self) -> bool:
        if self.action_count >= self.max_actions:
            return True
        return self.psg.solved()

    @property
    def observation(self) -> SynthEnvObservation:
        return SynthEnvObservation(
            psg=self.psg,
            action_count=self.action_count,
        )

    def step(self, action: SynthAction):
        """
        (1) Apply the action
        (2) Update environment's state

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        op, arg_nodes = action
        reward = op.apply_op(self.psg, arg_nodes)

        # self.psg.draw()

        self.action_count += 1
        if self.action_count == self.max_actions:
            reward = self.timeout_penalty

        return self.observation, reward, self.done, dict()
