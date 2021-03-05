import numpy as np
import random
import mlflow
from typing import Dict, Any

from bidir.task_utils import arc_task, twenty_four_task
from bidir.utils import load_mlflow_model
from rl.agent import ManualAgent, RandomAgent, SynthAgent
from rl.environment import SynthEnv
import rl.ops.arc_ops
from rl.random_programs import depth_one_random_sample
from rl.policy_net import policy_net_24
from experiments.supervised_training import DepthOneSampleDataset
import experiments.pol_grad_24

np.random.seed(3)
random.seed(3)


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

        action = agent.choose_action(env.observation())
        op, args = action
        state, reward, done, _ = env.step(action)

        # nodes = env.psg.get_value_nodes()
        # grounded_values = [g.value[0] for g in nodes if env.psg.is_grounded(g)]
        # ungrounded_values = [
        #     g.value[0] for g in nodes if not env.psg.is_grounded(g)
        # ]
        # if reward >= 0:
        #     print(f"op: {op.name}, args: {args}")
        # s = input()
        # if i % 10 == 0:
        # print(f"{i} actions attempted")
        i += 1

    if env.is_solved():
        print(f"We solved the task in {i} actions")
        print(f"program: {env.psg.get_program()}")
    else:
        print("We timed out during solving the task.")


def arc_manual_agent():
    task = arc_task(288, train=True)
    env = SynthEnv(task=task, ops=rl.ops.arc_ops.ALL_OPS, max_actions=100)
    agent = ManualAgent(rl.ops.arc_ops.ALL_OPS)

    run_until_done(agent, env)


def twenty_four_manual_agent(numbers):
    task = twenty_four_task(numbers, 24)
    env = SynthEnv(task=task, ops=rl.ops.twenty_four_ops.ALL_OPS)
    agent = ManualAgent(rl.ops.twenty_four_ops.ALL_OPS)

    run_until_done(agent, env)


def twenty_four_random_agent(numbers):
    # num_actions = []
    # program_lengths = []
    # for _ in range(20000):
    # if _ % 1000 == 0:
    # print(_)
    task = twenty_four_task(numbers, 24)
    OPS = rl.ops.twenty_four_ops.ALL_OPS
    env = SynthEnv(task=task, ops=OPS, max_actions=1000)
    agent = RandomAgent(OPS)

    done = False
    i = 0
    while not done:
        action = agent.choose_action(env.observation())
        op, args = action
        state, reward, done, _ = env.step(action)
        i += 1

    print(f"number of actions: {i}")
    program_str = str(env.psg.get_program())
    print(f"program: {program_str}")
    # program_len = 1 + program_str.count('(')
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

    ops = [rl.ops.arc_ops.OP_DICT[s] for s in op_strs]

    success = 0
    for i in range(1, 2):
        # print('trying again')
        task = arc_task(task_num, train=True)
        env = SynthEnv(task=task, ops=ops, max_actions=100)
        agent = RandomAgent(ops)

        run_until_done(agent, env)
        prog = env.psg.get_program()
        succeeded = prog is not None
        success += succeeded
        if i % 10 == 0:
            print('success ratio: ' + str(success / i))


def training():
    mlflow.set_experiment("Supervised training")
    data_size = 1000
    num_inputs = 2
    max_input_int = 10
    max_int = rl.ops.twenty_four_ops.MAX_INT
    enforce_unique = False
    # num_ops = 5
    # model_load_run_id = "ed7c161b2087492b93fa9a2943c34653"
    model_load_run_id = None
    save_model = False

    with mlflow.start_run():
        PARAMS = dict(
            data_size=data_size,
            num_inputs=num_inputs,
            max_input_int=max_input_int,
            max_int=max_int,
            enforce_unique=enforce_unique,
            # num_ops=num_ops,
            model_load_run_id=model_load_run_id,
            save_model=save_model,
        )

        mlflow.log_params(PARAMS)

        # ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:num_ops]
        ops = rl.ops.twenty_four_ops.FORWARD_OPS

        def spec_sampler():
            return depth_one_random_sample(ops,
                                           num_inputs=num_inputs,
                                           max_input_int=max_input_int,
                                           max_int=max_int,
                                           enforce_unique=enforce_unique)

        data = DepthOneSampleDataset(
            size=data_size,
            sampler=spec_sampler,
            fixed_set=False,
        )

        print('Preview of data points:')
        for i in range(min(10, len(data))):
            print(data.__getitem__(i))  # simply calling data[i] doesn't work

        if model_load_run_id:
            net = load_mlflow_model(model_load_run_id)
        else:
            net = policy_net_24(ops, max_int=max_int, state_dim=512)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

        print(f"number of parameters in model: {count_parameters(net)}")

        print(f"Starting run:\n{mlflow.active_run().info.run_id}")
        experiments.supervised_training.train(
            net,
            data,
            epochs=500,
            print_every=1,
            save_model=save_model,
        )

    # PG fine-tuning
    mlflow.set_experiment("Policy gradient")

    # hopefully this starts a new run?
    with mlflow.start_run():

        def task_sampler():
            return spec_sampler().task

        TRAIN_PARAMS = dict(
            discount_factor=0.5,
            epochs=50000,
            max_actions=10,
            batch_size=1000,
            lr=0.001,
            # mlflow can't log long lists
            # ops=OPS,
            reward_type=None,
            print_rewards_by_task=False,
            save_model=True,
        )

        AUX_PARAMS: Dict[str, Any] = dict(
            # model_load_path=load_path,
        )

        mlflow.log_params(TRAIN_PARAMS)
        mlflow.log_params(AUX_PARAMS)

        experiments.pol_grad_24.train(
            task_sampler=task_sampler,
            ops=ops,
            policy_net=net,
            **TRAIN_PARAMS,  # type: ignore
        )


if __name__ == '__main__':
    training()
    # simon_pol_grad()
    # rl.supervised_training.main()
    # rl.train_24_policy_old.main()
    # arc_manual_agent()
    # twenty_four_manual_agent()
