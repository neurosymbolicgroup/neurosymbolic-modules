import argparse
import numpy as np
import os
import random
import mlflow
from typing import Dict, Any, List, Tuple, Callable, Sequence
from multiprocessing import Pool
import uuid

from bidir.task_utils import arc_task, twenty_four_task, Task, binary_task
import bidir.utils as utils
from rl.agent import ManualAgent, RandomAgent, SynthAgent
from rl.environment import SynthEnv
import rl.ops.arc_ops
from rl.test_search import policy_rollouts
from rl.ops.operations import Op
import rl.ops.twenty_four_ops
import rl.ops.binary_ops
import rl.ops.binary_ops as binary_ops
# from rl.ops.operations import ForwardOp
import rl.random_programs as random_programs
import rl.data_analytics as data_analytics
import rl.policy_net as policy_net
from experiments.supervised_training import ActionDataset, program_dataset
import experiments.supervised_training as sv_train
import experiments.policy_gradient as pg
import torch

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
        #     print(f'{i}:\t({ground_string}){type(val.value[0])})\t{str(val)}')

        action = agent.choose_action(env.observation())
        op, args = action
        state, reward, done, _ = env.step(action)

        # nodes = env.psg.get_value_nodes()
        # grounded_values = [g.value[0] for g in nodes
        #                    if env.psg.is_grounded(g)]
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
    task = arc_task(379, train=True)
    env = SynthEnv(task=task, ops=rl.ops.arc_ops.ALL_OPS)
    agent = ManualAgent(rl.ops.arc_ops.ALL_OPS)

    run_until_done(agent, env)


def binary_manual_agent():
    task = binary_task(15)
    env = SynthEnv(task=task, ops=rl.ops.binary_ops.ALL_OPS)
    agent = ManualAgent(rl.ops.binary_ops.ALL_OPS)

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


def peter_john_demo():
    # first, need to generate a local dataset. May take some time.
    parallel_arc_dataset_gen()
    # do supervised pretraining. Then fine-tune model on darpa tasks.
    arc_training()

    # to evaluate "test-time", need to modify the rollouts method to load the
    # model that was just training, and then call it.
    rollouts()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def arc_task_sampler(ops,
                     inputs_small: bool = False,
                     depth: int = 3):

    if inputs_small:
        inputs_sampler = random_programs.random_arc_small_grid_inputs_sampler
    else:
        inputs_sampler = random_programs.random_arc_grid_inputs_sampler

    def sampler():
        # return data.random_sample().task
        inputs = inputs_sampler()
        program_spec = random_programs.random_bidir_program(ops=ops,
                                                            inputs=inputs,
                                                            depth=depth,
                                                            forward_only=True)
        return program_spec.task

    return sampler


def arc_training():
    # multithreading doesn't seem to help, so disable
    torch.set_num_threads(1)

    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    # if None, loads a new model
    model_load_run_id = None
    model_load_run_id = "0b8e182683d34ba4890f0baa4c9af857"  # fw sv mixed
    model_load_run_id = "a26d0aab8c1b402e8179356e6cf64e3f"  # fw sv mixed continue PG
    # model_load_run_id = "3e842a3acbf648779e82ef22c1dff0ce"  # bidir sv mixed
    model_load_name = 'model'
    # model_load_name = 'epoch-1499'

    # run_supervised = True; run_policy_gradient = False
    run_policy_gradient = True; run_supervised = False

    description = "Continue PG that was accidentally stopped"

    SUPERVISED_PARAMS = dict(
        save_every=1000,
        epochs=10000,
        lr=0.002,  # default: 0.002
        print_every=1,
        use_cuda=use_cuda,
        test_every=100,
    )

    PG_TRAIN_PARAMS = dict(
        epochs=10000,
        max_actions=5,
        batch_size=5000,
        lr=1e-5,  # default: 0.001
        reward_type='shaped',
        save_every=200,
        forward_only=True,
        use_cuda=use_cuda,
        entropy_ratio=0.0,
        test_every=100,
    )

    max_nodes = 25  # shouldn't change this between instantiations

    ops = rl.ops.arc_ops.BIDIR_GRID_OPS

    op_str = str(map(str, ops))
    op_str = op_str[0:min(249, len(op_str))]

    # saved by supervised too
    AUX_PARAMS: Dict[str, Any] = dict(
        description=description,
        model_load_run_id=model_load_run_id,
        model_name=model_load_name,
        ops=op_str,
        # for arc sampler
        small_inputs=True,
        depth=5,
        mixed_data=True,
        forward_only=PG_TRAIN_PARAMS['forward_only'],
    )

    if AUX_PARAMS['mixed_data']:
        if PG_TRAIN_PARAMS['forward_only']:
            data = arc_forward_supervised_data()
        else:
            data = arc_bidir_supervised_data()
    else:
        data = arc_dataset(PG_TRAIN_PARAMS['forward_only'],  # type: ignore
                           AUX_PARAMS['small_inputs'],
                           AUX_PARAMS['depth'])

    darpa_tasks, _ = arc_darpa_tasks()
    def darpa_sampler():
        return random.choice(darpa_tasks)

    arc_sampler = arc_task_sampler(
        ops,
        inputs_small=AUX_PARAMS['small_inputs'],
        depth=AUX_PARAMS['depth']
    )

    def dataset_sampler():
        return data.random_sample().task

    # policy_gradient_sampler = arc_sampler
    policy_gradient_sampler = dataset_sampler

    if model_load_run_id:
        net = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)
    else:
        net = policy_net.policy_net_arc(ops=ops, max_nodes=max_nodes, use_cuda=use_cuda)  # type: ignore
        print('Starting model from scratch')
    print(f"Number of parameters in model: {count_parameters(net)}")

    test_tasks, _ = arc_darpa_tasks()

    def rollout_fn():
        return policy_rollouts(net,
                               ops=ops,
                               tasks=test_tasks,
                               timeout=60,
                               max_actions=PG_TRAIN_PARAMS['max_actions'],
                               verbose=False)

    SUPERVISED_PARAMS['rollout_fn'] = rollout_fn
    PG_TRAIN_PARAMS['rollout_fn'] = rollout_fn

    if run_supervised:
        supervised_training(net, data, SUPERVISED_PARAMS, AUX_PARAMS,
                            experiment_name='Supervised training 2')
    if run_policy_gradient:
        policy_gradient(net, policy_gradient_sampler, ops, PG_TRAIN_PARAMS,
                        AUX_PARAMS, experiment_name='Policy gradient 2')


def supervised_training(net, data, params, aux_params,

                        experiment_name='Supervised training'):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_params(aux_params)
        mlflow.log_params(dict(id=mlflow.active_run().info.run_id))

        print(f"Starting run:\n{mlflow.active_run().info.run_id}")
        sv_train.train(
            net,
            data,
            **params,  # type: ignore
        )


def policy_gradient(net, task_sampler, ops, params, aux_params,
                    experiment_name='Policy gradient'):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_params(aux_params)
        mlflow.log_params(dict(id=mlflow.active_run().info.run_id))

        print(f"Starting run:\n{mlflow.active_run().info.run_id}")
        pg.train(
            task_sampler=task_sampler,
            ops=ops,
            policy_net=net,
            **params,  # type: ignore
        )


def binary_training():
    # multithreading doesn't seem to help, so disable
    torch.set_num_threads(1)

    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    model_load_run_id = None
    # model_load_run_id = "51eb0796c6f34a00ba338966b73589d6"  # bidir 24 mixed sv bidir

    model_load_name = 'model'
    # model_load_name = 'epoch-14000'

    run_supervised = True; run_policy_gradient = False
    # run_supervised = False; run_policy_gradient = True
    # run_supervised = True; run_policy_gradient = True

    forward_only = False
    description = "PG learning rate, batch size search."
    max_int = 5000000
    binary_ops.MAX_INT = max_int  # global variable..

    SUPERVISED_PARAMS = dict(
        save_every=1000,
        epochs=10000,
        lr=0.002,  # default: 0.002
        print_every=1,
        use_cuda=use_cuda,
        test_every=1,
    )

    PG_TRAIN_PARAMS = dict(
        epochs=20000,
        max_actions=binary_ops.max_actions(max_int),
        batch_size=5000,
        lr=1e-5,  # default: 0.001
        reward_type='shaped',
        print_every=100,
        save_every=100,
        use_cuda=use_cuda,
        forward_only=forward_only,
        entropy_ratio=0.0,
        test_every=100,
    )

    max_nodes = 10

    ops = rl.ops.binary_ops.ALL_OPS
    if forward_only:
        ops = rl.ops.binary_ops.FORWARD_OPS

    op_str = str(map(str, ops))
    op_str = op_str[0:min(249, len(op_str))]

    # saved by supervised too
    AUX_PARAMS: Dict[str, Any] = dict(
        description=description,
        model_load_run_id=model_load_run_id,
        max_int=max_int,
        held_out_ratio=0.1,
        data_size=5000,
        forward_only=forward_only,
        ops=op_str,
    )

    def policy_gradient_sampler():
        return binary_task(random.randint(2, AUX_PARAMS['max_int']))

    supervised_data, held_out_tasks = binary_supervised_data(
        max_int=AUX_PARAMS['max_int'],
        forward_only=forward_only,
        held_out_ratio=AUX_PARAMS['held_out_ratio'],
        data_size=AUX_PARAMS['data_size'])

    if model_load_run_id:
        net = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)
    else:
        net = policy_net.policy_net_binary2(
            ops,
            max_int=AUX_PARAMS['max_int'],
            state_dim=512,
            max_nodes=max_nodes,
            use_cuda=use_cuda)
        print('Starting model from scratch')
    print(f"Number of parameters in model: {count_parameters(net)}")

    test_tasks = held_out_tasks

    def rollout_fn():
        nonlocal test_tasks
        solved, attempted = policy_rollouts(net,
                                            ops=ops,
                                            tasks=test_tasks,
                                            timeout=20,
                                            max_actions=PG_TRAIN_PARAMS['max_actions'],
                                            verbose=False)
        solved_targets = [task.target[0] for task in solved]
        # print(sorted(solved_targets))
        # print(sorted([task.target[0] for task in test_tasks if task not in solved]))
        # print([binary_ops.program_length(t.target[0]) for t in sorted(test_tasks, key=lambda t: t.target[0]) if t not in solved])
        # print(len(solved_targets))
        # test_tasks = [t for t in test_tasks if t not in solved]
        return solved, attempted

    SUPERVISED_PARAMS['rollout_fn'] = rollout_fn
    PG_TRAIN_PARAMS['rollout_fn'] = rollout_fn

    if run_supervised:
        supervised_training(net, supervised_data, SUPERVISED_PARAMS, AUX_PARAMS,
                            experiment_name='24 Supervised Training')
    if run_policy_gradient:
        policy_gradient(net, policy_gradient_sampler, ops, PG_TRAIN_PARAMS,
                        AUX_PARAMS, experiment_name='24 Policy Gradient')


def binary_supervised_data(
    max_int: int,
    forward_only: bool = False,
    held_out_ratio: float = 0.05,
    data_size: int = 0,  # total: held out is taken from this
) -> Tuple[ActionDataset, List[Task]]:

    if data_size == 0 or data_size > max_int - 3:
        data_size = max_int - 3

    held_out_size = int(held_out_ratio * data_size)

    data_targets = list(random.sample(range(3, max_int + 1), k=data_size))

    held_out_targets = data_targets[-held_out_size:]
    held_out_tasks = [binary_task(i) for i in held_out_targets]
    data_targets = data_targets[:-held_out_size]

    programs = [binary_ops.make_binary_program(target, forward_only=forward_only)
                for target in data_targets]
    # for program in sorted(programs, key=lambda p: p.task.target[0]):
    #     print()
    #     print(f"task: {program.task}")
    #     for action_spec in program.action_specs:
    #         print(f"task2: {action_spec.task}")
    #         print(f"action: {action_spec.action}")

    return program_dataset(programs), held_out_tasks


def training_24():
    parser = argparse.ArgumentParser(description='training24')
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument("--forward_only",
                        action="store_true",
                        dest="forward_only")
    parser.add_argument("--mixed",
                        action="store_true",
                        dest="mixed")
    parser.add_argument("--entropy_ratio",
                        type=float,
                        default=0)

    args = parser.parse_args()

    # multithreading doesn't seem to help, so disable
    torch.set_num_threads(1)

    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    model_load_run_id = None
    model_load_run_id = "e0b94b6c83c741b0989679253a57fec3"

    model_load_name = 'model'
    # model_load_name = 'epoch-14000'

    # run_supervised = True; run_policy_gradient = False
    run_supervised = False; run_policy_gradient = True

    depth = args.depth
    # depth = 1  # 0 means mixed
    # forward_only = args.forward_only
    forward_only = True
    description = f"FW 24 mixed PG fine-tune depth {depth}"

    # if forward_only:
    #     if depth == 1:
    #         model_load_run_id = "07a54c05adad4735bc327f4aea072748"  # depth 1 supervised forward
    #     elif depth == 2:
    #         model_load_run_id = "3e879ca5a7cf4825adfac5d372ab13b3"  # depth 2 supervised forward
    #     elif depth == 3:
    #         model_load_run_id = "512562ac47eb439aaba1707556535e90"  # depth 3 supervised forward
    #     elif depth == 4:
    #         model_load_run_id = "242196ba97034e5bb8505a2217643990"  # depth 4 supervised forward

    #     if args.mixed:
    #         model_load_run_id = "864563ea4cdd42a592950c62804f96e7"  # mixed supervised forward
    # else:
    #     if depth == 1:
    #         model_load_run_id = "b1f7ca83d8ca40ba96a74ee1bda18993"  # depth 1 supervised bidir
    #     elif depth == 2:
    #         model_load_run_id = "78f5537acfce486b8f194a355cfc4858"  # depth 2 supervised bidir
    #     elif depth == 3:
    #         model_load_run_id = "fc5ff62e167c45449c9f40202abf9da0"  # depth 3 supervised bidir
    #     elif depth == 4:
    #         model_load_run_id = "365f7e17fa4b42bf876bf30450364a03"  # depth 4 supervised bidir

    #     if args.mixed:
    #         model_load_run_id = "51eb0796c6f34a00ba338966b73589d6"  # mixed supervised bidir

    SUPERVISED_PARAMS = dict(
        save_every=1000,
        epochs=15000,
        lr=0.002,  # default: 0.002
        print_every=100,
        use_cuda=use_cuda,
        test_every=500,
    )

    def max_actions(depth):
        if depth == 0:
            assert not run_policy_gradient
            return 6
        if depth == 1:
            return 1
        if depth == 2:
            return 4
        if depth == 3:
            return 5
        if depth == 4:
            return 6
        else:
            assert False, 'depth not account for yet'

    PG_TRAIN_PARAMS = dict(
        epochs=10000,
        max_actions=max_actions(depth),
        batch_size=1000,
        # lr=1e-5,  # default: 0.001
        lr=0.001,
        reward_type='shaped',
        print_every=100,
        save_every=2000,
        test_every=0,
        use_cuda=use_cuda,
        forward_only=forward_only,
        entropy_ratio=0,
    )

    max_nodes = 10

    # ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:3]
    ops = rl.ops.twenty_four_ops.ALL_OPS
    if forward_only:
        ops = rl.ops.twenty_four_ops.FORWARD_OPS

    op_str = str(map(str, ops))
    op_str = op_str[0:min(249, len(op_str))]

    # saved by supervised too
    AUX_PARAMS: Dict[str, Any] = dict(
        description=description,
        model_load_run_id=model_load_run_id,
        depth=depth,
        num_inputs=4,
        max_input_int=9,
        max_int=rl.ops.twenty_four_ops.MAX_INT,
        data_size=5000,
        forward_only=forward_only,
        ops=op_str,
    )

    def sampler():
        inputs = random.sample(range(1, AUX_PARAMS['max_input_int'] + 1),
                               k=AUX_PARAMS['num_inputs'])
        tuple_inputs = tuple((i,) for i in inputs)
        prog = random_programs.random_bidir_program(ops,
                                                    tuple_inputs,
                                                    AUX_PARAMS['depth'],
                                                    forward_only=True)
        return prog
        # return twenty_four_task((8, 7), )

    def policy_gradient_sampler():
        return sampler().task

    supervised_data = twenty_four_supervised_data(
        depth=AUX_PARAMS['depth'],
        forward_only=forward_only)

    if model_load_run_id:
        net = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)
    else:
        net = policy_net.policy_net_24_alt(ops,
                                           max_int=AUX_PARAMS['max_int'],
                                           state_dim=512,
                                           max_nodes=max_nodes,
                                           use_cuda=use_cuda)
        print('Starting model from scratch')
    print(f"Number of parameters in model: {count_parameters(net)}")

    test_tasks = twenty_four_test_tasks()

    def rollout_fn():
        return policy_rollouts(net,
                               ops=ops,
                               tasks=test_tasks,
                               timeout=60,
                               max_actions=6,
                               verbose=False)

    SUPERVISED_PARAMS['rollout_fn'] = rollout_fn
    PG_TRAIN_PARAMS['rollout_fn'] = rollout_fn

    if run_supervised:
        supervised_training(net, supervised_data, SUPERVISED_PARAMS, AUX_PARAMS,
                            experiment_name='24 Supervised Training')
    if run_policy_gradient:
        policy_gradient(net, policy_gradient_sampler, ops, PG_TRAIN_PARAMS,
                        AUX_PARAMS, experiment_name='24 Policy Gradient')


def arc_darpa_tasks() -> Tuple[List[Task], Dict[int, Task]]:
    task_nums = [
        82, 86, 105, 115, 139, 141, 149, 151, 154, 163, 171, 178, 209, 210,
        240, 248, 310, 379,
    ]
    tasks = [arc_task(task_num) for task_num in task_nums]
    task_dict = dict(zip(tasks, task_nums))
    return tasks, task_dict


def hard_arc_darpa_tasks():
    hard_tasks = [arc_task(task_num) for task_num in [82, 105, 141, 151, 210]]
    hard_tasks2 = []
    for task in hard_tasks:

        def i_th_in_front(tupl):
            return tuple([tupl[i], *tupl])

        for i in range(len(task.target)):
            # make i'th example in front, so policy net sees it
            task2 = Task(tuple(i_th_in_front(inp) for inp in task.inputs),
                         i_th_in_front(task.target))
            hard_tasks2.append(task2)

    return hard_tasks2


def twenty_four_test_tasks():
    twenty_four_nums = [
        9551, 9521, 8851, 8842, 8754, 8664, 8643, 8622, 8621, 8521, 8432, 8411,
        7773, 7764, 7543, 7411, 6666, 6621, 6521, 6511, 6422, 5543, 4331, 4321,
        9887, 9884, 9883, 9854, 9761, 9741, 9654, 9642, 9632, 9611, 9532, 9441,
        8874, 8872, 8844, 8821, 8768, 8732, 8731, 8722, 8665, 8655, 8653, 8652,
        8531, 8441, 8422, 7754, 7741, 7732, 7642, 7621, 7542, 7541, 7531, 7443,
        7432, 7422, 7332, 7322, 6651, 6631, 6543, 6522, 6442, 6432, 5433, 5432,
        9885, 9862, 9844, 9832, 9743, 9742, 9622, 9421, 8875, 8852, 8831, 8744,
        8633, 8552, 8522, 7641, 7533, 7532, 7431, 7331, 6641, 5442, 5333, 5322
    ]
    # 9951 -> (9, 9, 5, 1) for each example
    twenty_four_inputs = [tuple(map(int, str(n))) for n in twenty_four_nums]
    twenty_four_tasks = [twenty_four_task(i, 24) for i in twenty_four_inputs]
    return twenty_four_tasks


def rollouts():
    torch.set_num_threads(1)

    # model_load_run_id = "0b8e182683d34ba4890f0baa4c9af857"  # fw sv mixed
    model_load_run_id = "3e842a3acbf648779e82ef22c1dff0ce"  # bidir sv mixed
    model_load_name = 'model'

    model = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)

    # make sure using same ops that the net was trained on!
    ops = rl.ops.arc_ops.BIDIR_GRID_OPS

    test_tasks, task_dict = arc_darpa_tasks()

    # test_tasks = hard_arc_darpa_tasks()

    # note: if doing "for real", might want to change timeout time to do search
    # for much longer.
    timeout_seconds = 1800

    solved_tasks, attempts_per_task = policy_rollouts(model=model,
                                                      ops=ops,
                                                      tasks=test_tasks,
                                                      timeout=timeout_seconds,
                                                      max_actions=10)
    for task in sorted(solved_tasks, key=lambda t: attempts_per_task[t]):
        print(f"task {task_dict[task]} solved in {attempts_per_task[task]} attempts")
    # print(f"solved_tasks: {solved_tasks}")
    # print(f"attempts_per_task: {attempts_per_task}")
    # print(f"solved_tasks: {len(solved_tasks)}")


def get_24_policy_gradient_models(src_dir='out/') -> List[Tuple[str, str]]:
    paths = os.listdir(src_dir)

    out = []
    for path in paths:
        lines = data_analytics.read_lines(src_dir + '/' + path)
        assert data_analytics.is_completed(lines), f"path {path} not completed"
        if data_analytics.is_completed(lines):  # and path.endswith('_1.out'):
            model_id = data_analytics.get_model_id(lines)
            model = utils.load_mlflow_model(model_id)
            forward_only = 'bidir' not in path
            if forward_only:
                if len(model.ops) != 4:
                    print(f"path {path} not fw={forward_only}")
                    continue
            else:
                if len(model.ops) != 12:
                    print(f"path {path} not fw={forward_only}")
                    continue

            out.append((model_id, path))

    return sorted(out, key=lambda t: t[1])


def get_24_supervised_models() -> List[Tuple[str, str]]:
    return [
        ("07a54c05adad4735bc327f4aea072748", "depth 1 supervised forward"),
        ("3e879ca5a7cf4825adfac5d372ab13b3", "depth 2 supervised forward"),
        ("512562ac47eb439aaba1707556535e90", "depth 3 supervised forward"),
        ("242196ba97034e5bb8505a2217643990", "depth 4 supervised forward"),
        ("864563ea4cdd42a592950c62804f96e7", "mixed supervised forward"),
        ("b1f7ca83d8ca40ba96a74ee1bda18993", "depth 1 supervised bidir"),
        ("78f5537acfce486b8f194a355cfc4858", "depth 2 supervised bidir"),
        ("fc5ff62e167c45449c9f40202abf9da0", "depth 3 supervised bidir"),
        ("365f7e17fa4b42bf876bf30450364a03", "depth 4 supervised bidir"),
        ("51eb0796c6f34a00ba338966b73589d6", "mixed supervised bidir"),
    ]


def check_24_rollouts():
    models = get_24_supervised_models() + get_24_policy_gradient_models()
    for model_load_run_id, description in models:
        forward_only = 'bidir' not in description
        model = utils.load_mlflow_model(model_load_run_id)
        if forward_only:
            assert len(model.ops) == 4, f'{description} is lying'
        else:
            assert len(model.ops) == 12, f'{description} is lying'


def parallel_24_rollouts():
    args = get_24_supervised_models() + get_24_policy_gradient_models()

    for arg in args:
        run_24_rollouts2(arg)
    # with Pool() as p:
        # p.map(run_24_rollouts2, args)


def run_24_rollouts2(args):
    # need single arg to parallelize
    model_load_run_id, description = args
    forward_only = 'bidir' not in description

    solved_tasks = run_24_rollouts(model_load_run_id,
                                   forward_only=forward_only)
    print(description)
    print(model_load_run_id)
    print(len(solved_tasks))


def run_24_rollouts(model_load_run_id, model_load_name='model', forward_only: bool = False, timeout=60):
    torch.set_num_threads(1)

    max_actions = 5
    model = utils.load_mlflow_model(model_load_run_id, model_name=model_load_name)
    # make sure using same ops that the net was trained on!
    # ops = rl.ops.arc_ops.BIDIR_GRID_OPS
    ops = rl.ops.twenty_four_ops.ALL_OPS
    if forward_only:
        ops = rl.ops.twenty_four_ops.FORWARD_OPS

    # model = policy_net.policy_net_24_alt(ops,
    #                           max_int=100,
    #                           state_dim=512,
    #                           max_nodes=10,
    #                           use_cuda=False)

    test_tasks = twenty_four_test_tasks()

    max_inputs = max(len(task.inputs) for task in test_tasks)
    model.max_nodes = max_actions + max_inputs + 1

    solved_tasks, attempts_per_task = policy_rollouts(model=model,
                                                      ops=ops,
                                                      tasks=test_tasks,
                                                      timeout=timeout,
                                                      max_actions=max_actions,
                                                      verbose=False)
    # print(f"solved_tasks: {solved_tasks}")
    # print(f"attempts_per_task: {attempts_per_task}")
    print(f"solved {len(solved_tasks)} tasks out of {len(test_tasks)}")
    return solved_tasks


def bidir_dataset(path: str,
                  depth: int,
                  ops: Sequence[Op],
                  inputs_sampler: Callable[[], Tuple[Tuple[Any, ...],
                                                     ...]],
                  num_samples: int,
                  forward_only: bool = False):
    '''
    Generates a dataset of random arc programs. This is done separately from
    training because generating samples is rather expensive, so it slows down
    the training a lot. This doesn't make 100% of sense to simon, because it
    seems like training the NN should be taking way longer, but for now,
    whatever.

    This one is for bidirectional supervised examples.
    '''

    if not os.path.exists(path):
        os.mkdir(path)

    print(f'Generating examples of depth {depth}')

    while len(os.listdir(path)) < num_samples:
        # since each program gives multiple examples
        if len(os.listdir(path)) % 1000 < depth:
            print(f'Reached {len(os.listdir(path))} examples..')

        inputs = inputs_sampler()
        program_spec = random_programs.random_bidir_program(ops=ops,
                                                            inputs=inputs,
                                                            depth=depth,
                                                            forward_only=forward_only)
        # print(f'program: {program_spec.task}')
        # for action_spec in program_spec.action_specs:
        #     action = action_spec.action
        #     op_name = ops[action.op_idx].name
        #     print(f"op_name: {op_name}")
        #     arg_idxs = action.arg_idxs
        #     print(f"arg_idxs: {arg_idxs}")

        for action_spec in program_spec.action_specs:
            # print(f"action_spec: {action_spec}")
            filepath = path + '/' + str(uuid.uuid4())
            utils.save_action_spec(action_spec, filepath)


def arc_dataset(forward_only: bool, small_inputs: bool, depth: int):
    s = ('data/bidir/arc_'
         + ('fw_only_' if forward_only else 'bidir_')
         + ('small_' if small_inputs else '')
         + f"depth{depth}")
    return sv_train.ActionDatasetOnDisk(s)


def arc_forward_supervised_data() -> sv_train.ActionDatasetOnDisk2:
    return sv_train.ActionDatasetOnDisk2(dirs=[
        'data/bidir/arc_fw_only_depth3',
        'data/bidir/arc_fw_only_depth5',
        'data/bidir/arc_fw_only_depth10',
        'data/bidir/arc_fw_only_depth20',
        'data/bidir/arc_fw_only_small_depth3',
        'data/bidir/arc_fw_only_small_depth5',
        'data/bidir/arc_fw_only_small_depth10',
        'data/bidir/arc_fw_only_small_depth20',
    ])


def arc_bidir_supervised_data() -> sv_train.ActionDatasetOnDisk2:
    return sv_train.ActionDatasetOnDisk2(dirs=[
        'data/bidir/arc_bidir_depth3',
        'data/bidir/arc_bidir_depth5',
        'data/bidir/arc_bidir_depth10',
        'data/bidir/arc_bidir_depth20',
        'data/bidir/arc_bidir_small_depth3',
        'data/bidir/arc_bidir_small_depth5',
        'data/bidir/arc_bidir_small_depth10',
        'data/bidir/arc_bidir_small_depth20',
    ])


def twenty_four_supervised_data(depth=None, forward_only=True):
    if forward_only:
        if not depth:
            return sv_train.ActionDatasetOnDisk2(dirs=[
                'data/bidir/24_fw_only_depth1',
                'data/bidir/24_fw_only_depth2',
                'data/bidir/24_fw_only_depth3',
                'data/bidir/24_fw_only_depth4',
            ])
        else:
            return sv_train.ActionDatasetOnDisk(
                directory=f'data/bidir/24_fw_only_depth{depth}')
    else:
        if not depth:
            return sv_train.ActionDatasetOnDisk2(dirs=[
                'data/bidir/24_bidir_depth1',
                'data/bidir/24_bidir_depth2',
                'data/bidir/24_bidir_depth3',
                'data/bidir/24_bidir_depth4',
            ])
        else:
            return sv_train.ActionDatasetOnDisk(
                directory=f'data/bidir/24_bidir_depth{depth}')


def twenty_four_bidir_dataset_gen(args: Tuple):
    # args has to be unreadable like this so that f takes single arg, so that
    # it can be parallelized more easily
    # could make it a dict though
    depth, forward_only = args

    def sampler():
        return random_programs.random_twenty_four_inputs_sampler(max_input_int=9,
                                                                 num_inputs=4)
    if forward_only:
        path = f"data/bidir/24_fw_only_depth{depth}"
    else:
        path = f"data/bidir/24_bidir_depth{depth}"

    ops = rl.ops.twenty_four_ops.ALL_OPS
    bidir_dataset(path,
                  depth=depth,
                  inputs_sampler=sampler,
                  num_samples=50000,
                  forward_only=forward_only,
                  ops=ops)


def arc_bidir_dataset_gen(args: Tuple):
    # args has to be unreadable like this so that f takes single arg, so that
    # it can be parallelized more easily
    depth, forward_only, small = args
    if small:
        inputs_sampler = random_programs.random_arc_small_grid_inputs_sampler
        if forward_only:
            path = f"data/bidir/arc_fw_only_small_depth{depth}"
        else:
            path = f"data/bidir/arc_bidir_small_depth{depth}"
    else:
        inputs_sampler = random_programs.random_arc_grid_inputs_sampler
        if forward_only:
            path = f"data/bidir/arc_fw_only_depth{depth}"
        else:
            path = f"data/bidir/arc_bidir_depth{depth}"

    ops = rl.ops.arc_ops.BIDIR_GRID_OPS

    bidir_dataset(path,
                  depth=depth,
                  inputs_sampler=inputs_sampler,
                  num_samples=10000,
                  forward_only=forward_only,
                  ops=ops)


def parallel_arc_dataset_gen():
    args = [(depth, forward_only, small) for depth in [3, 5, 10, 20]
            for forward_only in [True, False] for small in [True, False]]

    with Pool() as p:
        p.map(arc_bidir_dataset_gen, args)
    # for arg in args:
        # arc_bidir_dataset_gen(arg)


def parallel_24_dataset_gen():
    args = [(depth, forward_only) for depth in [1, 2, 3]
            for forward_only in [True, False]]

    with Pool() as p:
        p.map(twenty_four_bidir_dataset_gen, args)

    # if unparallel desired
    # for arg in args:
        # arc_bidir_dataset_gen(arg)


def bidir_arc_heldout_tasks():
    grids = get_arc_grids()  # cached, so ok to call repeatedly
    small_grids = [g for g in grids if g.arr.shape == (3, 3)]

    def grid_sampler():
        return random.sample(small_grids)


if __name__ == '__main__':
    random.seed(45)
    torch.manual_seed(45)

    # binary_training()
    # arc_training()
    training_24()

    # arc_manual_agent()

    # rollouts()

    # parallel_24_dataset_gen()
    # parallel_24_rollouts()
    # rollouts_24()

    # check_24_rollouts()
    # get_24_paths()

    # peter_john_demo()
    # parallel_24_dataset_gen()
