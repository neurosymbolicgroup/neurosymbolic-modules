from typing import Any, NamedTuple, Sequence, Tuple

import gym

from bidir.utils import SynthError
from rl.ops.operations import Op
from rl.program_search_graph import ProgramSearchGraph, ValueNode


class SynthEnvAction(NamedTuple):
    op_idx: int
    arg_idxs: Tuple[int, ...]


class SynthEnvObservation(NamedTuple):
    psg: ProgramSearchGraph

    # Hidden state for debugging agents
    action_count_: int


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
    metadata = {"render.modes": ["matplotlib", "text"]}

    def __init__(
        self,
        # each example is a tuple (input, output)
        train_examples: Tuple[Tuple[Tuple, Any], ...],
        test_examples: Tuple[Tuple[Tuple, Any], ...],
        ops: Sequence[Op],
        max_actions=100,
        solve_reward=100,
        synth_error_penalty=-1,
        timeout_penalty=0,
    ):
        """
        Initialize the environment for given set of training and test examples.

        Each example is a tuple (input, output)
        If 'input' is a tuple, then the example is interpreted to have
        multiple inputs. Otherwise, we assume there's only one input

        All ops are provided at the beginning as well.
        """
        self.ops = ops
        self.max_actions = max_actions

        self.solve_reward = solve_reward
        self.synth_error_penalty = synth_error_penalty
        self.timeout_penalty = timeout_penalty

        num_in_args = len(train_examples[0][0])
        assert all(
            len(in_args) == num_in_args for (in_args, _) in train_examples)
        self.in_values = tuple(
            tuple(in_args[idx] for (in_args, _) in train_examples)
            for idx in range(num_in_args))

        self.out_values = tuple(out_val for (_, out_val) in train_examples)

        self.reset()

    def reset(self) -> SynthEnvObservation:
        self.action_count = 0
        self.psg = ProgramSearchGraph(self.in_values, self.out_values)
        return self.observation()

    def observation(self) -> SynthEnvObservation:
        return SynthEnvObservation(
            psg=self.psg,
            action_count_=self.action_count,
        )

    def render(self, mode="text"):
        if mode == "text":
            print(f"Solved: {self.is_solved()}")
            print(f"Number of actions: {self.action_count}")
            print(f"Number of nodes: {len(self.psg.get_value_nodes())}")
        elif mode == "matplotlib":
            self.psg.draw()
        else:
            raise NotImplementedError

    def done(self) -> bool:
        if self.max_actions != -1 and self.action_count >= self.max_actions:
            return True
        return self.psg.solved()

    def is_solved(self) -> bool:
        return self.psg.solved()

    def step(
        self, action: SynthEnvAction
    ) -> Tuple[SynthEnvObservation, float, bool, dict]:
        """
        (1) Apply the action
        (2) Update environment's state

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        reward = 0

        try:
            op = self.ops[action.op_idx]
            nodes = self.psg.get_value_nodes()
            arg_nodes = tuple(nodes[arg_idx] for arg_idx in action.arg_idxs)
            op.apply_op(self.psg, arg_nodes)
        except SynthError:
            # this covers a lot of possible errors:
            # 1. args to op's fn cause a syntax/type error
            # 2. args to forward op aren't grounded, etc.
            # 3. forward op creates a value that already exists and is
            #    grounded, etc.
            reward = self.synth_error_penalty

        if self.psg.solved():
            reward = self.solve_reward

        self.action_count += 1
        if self.action_count == self.max_actions:
            reward = self.timeout_penalty

        return self.observation(), reward, self.done(), dict()
