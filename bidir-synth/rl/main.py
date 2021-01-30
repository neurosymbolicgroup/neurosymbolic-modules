"""
This file should be run from the ec directory.
You can run this file by running `python -m rl.main`.
"""
import numpy as np
import random

from bidir.task_utils import get_task_examples
from bidir.utils import SynthError
from rl.agent import ManualAgent, RandomAgent, SynthAgent
from rl.arc_ops import OP_DICT, tuple_return
from rl.environment import SynthEnv
from rl.operations import ForwardOp, InverseOp
from rl.program_search_graph import ProgramSearchGraph, ValueNode

np.random.seed(3)
random.seed(3)

def run_until_done(agent: SynthAgent, env: SynthEnv):
    """
    Basic sketch of running an agent on the environment.
    There may be ways of doing this which uses the gym framework better.
    Seems like there should be a way of modularizing the choice of RL
    algorithm, the agent choice, and the environment choice.
    """

    MAXITERATIONS=5
    while (not env.done) and (env.observation.action_count < MAXITERATIONS):
        # env.psg.draw()

        try:
            action = agent.choose_action(env.observation)
            print("Final action chosen:", action[0])#, "Args:", action[1])
            state, reward, done, _ = env.step(action)
        except SynthError:
            reward = -1 # or whatever we want the reward to be if the action raises an error

        print('Reward: {}'.format(reward))
        print()

    if (env.done):
        print("We solved the task.")
    else:
        print("We timed out during solving the task.")


def arcexample_forward():
    """
    An example showing how the state updates
    when we apply a single-argument function (rotate) in the forward direction
    """
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid

    start_grids = (
        Grid(np.array([[0, 0], [1, 1]])),
        Grid(np.array([[2, 2], [2, 2]])),
    )

    end_grids = (
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]])),
    )
    psg = ProgramSearchGraph(start_grids, end_grids)

    # psg.draw()

    op1 = ForwardOp(rotate_ccw)
    op1.apply_op(psg, (psg.start, ))

    op2 = ForwardOp(rotate_cw)
    op2.apply_op(psg, (psg.start, ))

    # psg.draw()


def arcexample_backward():
    """
    An example showing how the state updates
    when we apply a single-argument function (rotate) in the backward direction
    """
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid

    start_grids = (
        Grid(np.array([[0, 0], [1, 1]])),
        Grid(np.array([[2, 2], [2, 2]])),
    )

    end_grids = (
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]])),
    )
    psg = ProgramSearchGraph(start_grids, end_grids)

    # psg.draw()

    op = InverseOp(forward_fn=rotate_ccw, inverse_fn=tuple_return(rotate_cw))
    op.apply_op(psg, (psg.end, ))

    # psg.draw()


def arcexample_multiarg_forward():
    """
    An example showing how the state updates
    when we apply a multi-argument function (inflate) in the forward direction
    """
    from bidir.primitives.functions import inflate
    from bidir.primitives.types import Grid

    start_grids = (
        Grid(np.array([[0, 0], [0, 0]])),
        Grid(np.array([[1, 1], [1, 1]])),
    )

    end_grids = (
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]])),
    )
    psg = ProgramSearchGraph(start_grids, end_grids)

    # psg.draw()

    v2 = ValueNode((2, ) * psg.num_examples)
    psg.add_constant(v2)

    # psg.draw()

    # inflate_op = OP_DICT['inflate_cond_inv']
    # apply_forward_op(state, inflate_op, [state.start])
    op = ForwardOp(inflate)
    op.apply_op(psg, (psg.start, v2))
    # state.draw()


def run_manual_agent():
    train_exs, test_exs = get_task_examples(56, train=True)
    env = SynthEnv(train_exs, test_exs, max_actions=100)
    agent = ManualAgent(OP_DICT)

    run_until_done(agent, env)

def run_random_agent():
    train_exs, test_exs = get_task_examples(56, train=True)
    env = SynthEnv(train_exs, test_exs, max_actions=100)
    agent = RandomAgent(OP_DICT)

    run_until_done(agent, env)

if __name__ == '__main__':
    run_random_agent()
    # run_manual_agent()
    # arcexample_forward()
    # arcexample_backward()