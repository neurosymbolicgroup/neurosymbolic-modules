"""
This file should be run from the biir-synth directory.
You can run this file by running `python -m rl.main`.
"""
import numpy as np
import random

from typing import List
from bidir.task_utils import get_arc_task_examples
from bidir.utils import SynthError
from rl.agent import ManualAgent, RandomAgent, SynthAgent
from rl.arc_ops import OP_DICT, tuple_return
from bidir.twenty_four import OP_DICT as TWENTY_FOUR_OP_DICT
from rl.environment import SynthEnv
from rl.operations import ForwardOp, InverseOp, Op
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from rl.policy_net import PolicyNet24
import modules.test_networks as test_networks
import modules.train_24_policy as train_24_policy

# np.random.seed(3)
# random.seed(3)


def run_until_done(agent: SynthAgent, env: SynthEnv):
    """
    Basic sketch of running an agent on the environment.
    There may be ways of doing this which uses the gym framework better.
    Seems like there should be a way of modularizing the choice of RL
    algorithm, the agent choice, and the environment choice.
    """

    done = False
    i = 0
    while not done:
        # for i, val in enumerate(env.psg.get_value_nodes()):
        #     ground_string = "G" if env.psg.is_grounded(val) else "UG"
        #     print(f'{i}:\t({ground_string}) {type(val.value[0])})\t{str(val)}')

        action = agent.choose_action(env.observation)
        op, args = action
        state, reward, done, _ = env.step(action)

        nodes = env.psg.get_value_nodes()
        grounded_values = [g.value[0] for g in nodes if env.psg.is_grounded(g)]
        ungrounded_values = [g.value[0] for g in nodes
                             if not env.psg.is_grounded(g)]
        if reward >= 0:
            print(f"op: {op.name}, args: {args}")
        # s = input()
        # if i % 10 == 0:
        # print(f"{i} actions attempted")
        i += 1

    if env.was_solved:
        print(f"We solved the task in {i} actions")
        print(f"program: {env.psg.get_program()}")
    else:
        print("We timed out during solving the task.")


def run_arc_manual_agent():
    train_exs, test_exs = get_arc_task_examples(56, train=True)
    env = SynthEnv(train_exs, test_exs, max_actions=100)
    agent = ManualAgent(OP_DICT)

    run_until_done(agent, env)


def run_twenty_four_manual_agent(numbers):
    # numbers = tuple(StartInt(n) for n in numbers)
    train_exs = ((numbers, 24), )
    env = SynthEnv(train_exs, tuple())
    agent = ManualAgent(TWENTY_FOUR_OP_DICT)

    run_until_done(agent, env)


def test_policy_net():
    train_exs = (((1, 2, 3, 4), 0), )
    env = SynthEnv(train_exs, tuple())
    # agent = ManualAgent(TWENTY_FOUR_OP_DICT)
    pn = PolicyNet24(list(TWENTY_FOUR_OP_DICT.values()))
    pn(env.psg)


def test_training_nets():
    test_networks.generate_dataset()


def twenty_four_random_agent(numbers):
    # num_actions = []
    # program_lengths = []
    # for _ in range(20000):
        # if _ % 1000 == 0:
            # print(_)
    train_exs = ((numbers, 24), )
    env = SynthEnv(train_exs, tuple(), max_actions=1000)
    agent = RandomAgent(list(TWENTY_FOUR_OP_DICT.values()))

    done = False
    i = 0
    while not done:
        action = agent.choose_action(env.observation)
        op, args = action
        state, reward, done, _ = env.step(action)
        i += 1

    print(f"number of actions: {i}")
    program_str = str(env.psg.get_program())
    print(f"program: {program_str}")
    program_len = 1 + program_str.count('(')
    # num_actions.append(i)
    # program_lengths.append(program_len)

    # from matplotlib import pyplot as plt
    # plt.hist(num_actions, bins=50)
    # plt.xlabel('number of actions to find 24 program for inputs (2, 3, 4)')
    # plt.show()
    # plt.hist(program_lengths, bins=50)
    # plt.xlabel('length of 24 program discovered for inputs (2, 3, 4)')
    # plt.show()

def arc_random_agent():
    op_strs = ['block', '3', 'Color.RED', 'get_color']
    task_num = 128

    ops = [OP_DICT[s] for s in op_strs]

    success = 0
    for i in range(1, 2):
        # print('trying again')
        train_exs, test_exs = get_arc_task_examples(task_num, train=True)
        env = SynthEnv(train_exs, test_exs, max_actions=100)
        agent = RandomAgent(ops)

        run_until_done(agent, env)
        prog = env.psg.get_program()
        succeeded = prog is not None
        success += succeeded
        if i % 10 == 0:
            print('success ratio: ' + str(success / i))


if __name__ == '__main__':
    # test_training_nets()
    # test_policy_net()
    # train_24_policy.generate_dataset()
    train_24_policy.main()
    # random_stuff()
    # run_arc_manual_agent()
    # run_twenty_four_manual_agent((6, 4))
    # twenty_four_random_agent((11, 3, 6, 9))
