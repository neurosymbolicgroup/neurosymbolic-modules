import gym
from rl.operations import Op
from typing import Tuple, List
from bidir.primitives.types import Grid
from rl.state import State, ValueNode


class ArcEnvironment(gym.Env):
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
        ops: List[Op],
        max_actions=100,
    ):
        """
        Initialize the environment for given set of training and test examples.

        All ops are provided at the beginning as well.

        Maybe we should eventually allow for arbitrary synthesis tasks, not
        just Grid -> Grid tasks.
        """

        # currently only train examples supported
        # self.state = State(train_examples, test_examples)
        in_grids, out_grids = zip(*train_examples)
        self.state = State(in_grids, out_grids)
        self.ops = ops
        # number of args an op will take
        self.arity = max(op.arity for op in ops)
        self.max_actions = max_actions
        self.action_count = 0
        self.done = self.state.done()
        self.reward_if_max_actions_hit = -1

    def step(self, action: Tuple[Op, List[ValueNode]]):
        """
        (1) Apply the action
        (2) Update environment's state
        """
        op, arg_nodes = action
        assert len(arg_nodes) == self.arity
        reward = op.apply_op(self.state, arg_nodes)

        self.done = self.state.done()

        self.action_count += 1
        if self.action_count == self.max_actions:
            reward = self.reward_if_max_actions_hit
            self.done = True

        return self.state, reward, self.done

    def setup(self):
        """
        Set up initial state of environment.
        """
        pass
