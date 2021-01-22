import sys
# should be running from ec directory!
sys.path.append(".")
from rl.environment import ArcEnvironment
from rl.create_ops import OP_DICT
from bidir.task_utils import get_task_examples
from rl.agent import ManualAgent, ArcAgent

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
    print('done: {}'.format(done))
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        print('reward: {}'.format(reward))


def arcexample_multiarg_forward():
    """
    An example showing how the state updates
    when we apply a multi-argument function (inflate) in the forward direction
    """

    import sys; sys.path.append("..") # hack to make importing bidir work
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
    apply_forward_op(state,two_op,[state.start])

    print(state.graph.nodes)
    state.draw()

    # extend in the forward direction using fn and tuple of arguments that fn takes
    # print(state.graph.nodes)

    # inflate_op = OP_DICT['inflate_cond_inv']
    # apply_forward_op(state, inflate_op, [state.start])
    # state.draw()



def run_manual_agent():
    ops = [
        'rotate_cw',
        'rotate_ccw',
        'vstack_pair',
        'vstack_pair_cond_inv',  # have to ask for inverse op explicitly
        'hflip',
        'hflip_inv',
        'vflip',
    ]
    ops = [OP_DICT[op_name] for op_name in ops]

    # train_exs, test_exs = get_task_examples(86, train=True)
    train_exs, test_exs = get_task_examples(115, train=True)
    env = ArcEnvironment(train_exs, test_exs, ops, max_actions=100)
    agent = ManualAgent(ops, env.arity, OP_DICT)

    run_until_done(agent, env)


if __name__ == '__main__':
    run_manual_agent()
    # arcexample_forward()
    # arcexample_backward()
    # arcexample_multiarg_forward()


