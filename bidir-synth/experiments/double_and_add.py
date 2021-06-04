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
from rl.main import policy_gradient, supervised_training, count_parameters
from experiments.supervised_training import ActionDataset, program_dataset
import experiments.supervised_training as sv_train
import experiments.policy_gradient as pg
import torch


def binary_training():
    parser = argparse.ArgumentParser(description='binary_training')
    parser.add_argument("--forward_only",
                        action="store_true",
                        dest="forward_only")
    parser.add_argument("--mixed",
                        action="store_true",
                        dest="mixed")
    parser.add_argument("--epochs",
                        type=int,
                        default=30)

    args = parser.parse_args()
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

    forward_only = args.forward_only
    description = "PG learning rate, batch size search."
    max_int = 5000000
    binary_ops.MAX_INT = max_int  # global variable..

    SUPERVISED_PARAMS = dict(
        save_every=1000,
        epochs=args.epochs,
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
                                            # timeout=1,
                                            max_actions=PG_TRAIN_PARAMS['max_actions'],
                                            verbose=False)
        # solved_targets = [task.target[0] for task in solved]
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


if __name__ == '__main__':
    binary_training()
