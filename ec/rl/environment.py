from typing import NamedTuple, Tuple

import gym

from rl.operations import Op
from bidir.primitives.types import Grid
from rl.program_search_graph import ProgramSearchGraph, ValueNode


class ArcEnvObservation(NamedTuple):
    psg: ProgramSearchGraph
    action_count: int


class ArcEnv(gym.Env):
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
        train_examples: Tuple[Tuple[Grid, Grid], ...],
        test_examples: Tuple[Tuple[Grid, Grid], ...],
        max_actions=100,
    ):
        """
        Initialize the environment for given set of training and test examples.

        All ops are provided at the beginning as well.

        Maybe we should eventually allow for arbitrary synthesis tasks, not
        just Grid -> Grid tasks.
        """
        self.max_actions = max_actions
        self.action_count = 0
        self.reward_if_max_actions_hit = -1

        # currently only train examples supported
        in_grids, out_grids = zip(*train_examples)
        self.psg = ProgramSearchGraph(in_grids, out_grids)  # type: ignore

    @property
    def done(self) -> bool:
        if self.action_count >= self.max_actions:
            return True
        return self.psg.solved()

    @property
    def observation(self) -> ArcEnvObservation:
        return ArcEnvObservation(
            psg=self.psg,
            action_count=self.action_count,
        )

    def step(self, action: Tuple[Op, Tuple[ValueNode]]):
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
            reward = self.reward_if_max_actions_hit

        return self.observation, reward, self.done, dict()
