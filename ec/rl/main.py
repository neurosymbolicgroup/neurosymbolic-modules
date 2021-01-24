import sys
# should be running from ec directory!
sys.path.append(".")
from rl.environment import ArcEnvironment
from rl.create_ops import OP_DICT
from bidir.task_utils import get_task_examples
from rl.agent import ManualAgent, ArcAgent
from rl.state import State

import numpy as np


def run_until_done(agent: ArcAgent, env: ArcEnvironment):
    """
    Basic sketch of running an agent on the environment.
    There may be ways of doing this which uses the gym framework better.
    Seems like there should be a way of modularizing the choice of RL
    algorithm, the agent choice, and the environment choice.
    """
    state = env.state
    done = env.done
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        print('reward: {}'.format(reward))


def arcexample_forward():
    """
    An example showing how the state updates
    when we apply a single-argument function (rotate) in the forward direction
    """

    import sys; sys.path.append("..") # hack to make importing bidir work
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid
    from rl.create_ops import OP_DICT

    start_grids = [
        Grid(np.array([[0, 0], [1, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]

    end_grids = [
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    state = State(start_grids, end_grids)


    state.draw()

    # create operation
    rotate_ccw_func = Function("rotateccw", rotate_ccw, [Grid], [Grid])
    rotate_cw_func = Function("rotatecw", rotate_cw, [Grid], [Grid])
    op = Op(rotate_ccw_func, rotate_cw_func, 'forward')

    # extend in the forward direction using fn and tuple of arguments that fn takes
    apply_forward_op(state, op, [state.start])
    state.draw()


def arcexample_backward():
    """
    An example showing how the state updates
    when we apply a single-argument function (rotate) in the backward direction
    """

    import sys; sys.path.append("..") # hack to make importing bidir work
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid

    from actions import apply_inverse_op

    start_grids = [
        Grid(np.array([[0, 0], [1, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]

    end_grids = [
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    state = State(start_grids, end_grids)


    state.draw()

    # create operation
    rotate_ccw_func = Function("rotateccw", rotate_ccw, [Grid], [Grid])
    rotate_cw_func = Function("rotatecw", rotate_cw, [Grid], [Grid])
    op = Op(rotate_ccw_func, rotate_cw_func, 'inverse')

    # extend in the forward direction using fn and tuple of arguments that fn takes
    apply_inverse_op(state, op, state.end)
    state.draw()

def arcexample_multiarg_forward():
    """
    An example showing how the state updates
    when we apply a multi-argument function (inflate) in the forward direction
    """

    import sys
    sys.path.append("..")  # hack to make importing bidir work
    from bidir.primitives.functions import inflate, deflate
    from bidir.primitives.types import Grid

    from actions import apply_forward_op

    start_grids = [
        Grid(np.array([[0, 0], [0, 0]])),
        Grid(np.array([[1, 1], [1, 1]]))
    ]

    end_grids = [
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    state = State(start_grids, end_grids)

    # state.draw()

    # get the number 2 in there
    # we are officially taking its input argument as state.start
    #   just so we know how many training examples there are
    #   and so we know it will show up in the graph
    two_op = OP_DICT['2']
    apply_forward_op(state, two_op, [state.start])

    print(state.graph.nodes)
    state.draw()

    # extend in the forward direction using fn and tuple of arguments that fn takes
    # print(state.graph.nodes)

    # inflate_op = OP_DICT['inflate_cond_inv']
    # apply_forward_op(state, inflate_op, [state.start])
    # state.draw()


def run_manual_agent():
    train_exs, test_exs = get_task_examples(56, train=True)
    env = ArcEnvironment(train_exs, test_exs, max_actions=100)
    agent = ManualAgent(OP_DICT)

    run_until_done(agent, env)


if __name__ == '__main__':
    run_manual_agent()
    # arcexample_forward()
    # arcexample_backward()
    # arcexample_multiarg_forward()
