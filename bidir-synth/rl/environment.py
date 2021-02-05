from bidir.utils import SynthError
from typing import NamedTuple, Optional, Tuple, Any

import gym

from rl.operations import Op
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
        # each example is a tuple (input, output)
        train_examples: Tuple[Tuple[Any, Any], ...],
        test_examples: Tuple[Tuple[Any, Any], ...],
        max_actions=100,
    ):
        """
        Initialize the environment for given set of training and test examples.

        Each example is a tuple (input, output)
        If 'input' is a tuple, then the example is interpreted to have
        multiple inputs. Otherwise, we assume there's only one input

        All ops are provided at the beginning as well.

        """
        self.max_actions = max_actions
        self.action_count = 0
        self.timeout_penalty = 0
        self.solve_reward = 1
        self.synth_error_penalty = -1

        if not isinstance(train_examples[0][0], tuple):
            # single input. transform to tuplized version
            train_examples = [((ex[0],),ex[1]) for ex in train_examples]  # type: ignore
            test_examples = [((ex[0],),ex[1]) for ex in test_examples]  # type: ignore

        # currently only train examples supported
        # tuple of shape (num_examples, num_inputs)
        in_values: Tuple[Tuple[Any, ...], ...]
        # tuple of shape (num_examples)
        out_values: Tuple[Any, ...]
        in_values, out_values = zip(*train_examples)
        # go from (num_examples, num_inputs) to (num_inputs, num_examples)
        in_values = tuple(zip(*in_values))
        self.psg = ProgramSearchGraph(in_values, out_values)

    @property
    def done(self) -> bool:
        if self.max_actions != -1 and self.action_count >= self.max_actions:
            return True
        return self.psg.solved()

    @property
    def was_solved(self) -> bool:
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
        reward = 0

        try:
            op.apply_op(self.psg, arg_nodes)
        except SynthError:
            reward = self.synth_error_penalty

        if self.psg.solved():
            reward = self.solve_reward

        # self.psg.draw()

        self.action_count += 1
        if self.action_count == self.max_actions:
            reward = self.timeout_penalty

        return self.observation, reward, self.done, dict()
