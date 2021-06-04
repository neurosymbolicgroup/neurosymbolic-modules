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
import rl.policy_net as policy_net
from experiments.supervised_training import ActionDataset, program_dataset
import experiments.supervised_training as sv_train
import experiments.policy_gradient as pg
import torch

from rl.main import parallel_arc_dataset_gen, arc_forward_supervised_data, arc_dataset, arc_darpa_tasks, count_parameters, supervised_training, policy_gradient, arc_bidir_supervised_data, arc_task_sampler, arc_task_sampler2


def arc_training():
    parser = argparse.ArgumentParser(description='training24')
    parser.add_argument("--forward_only",
                        action="store_true",
                        dest="forward_only")

    args = parser.parse_args()

    # multithreading doesn't seem to help, so disable
    torch.set_num_threads(1)

    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    # if None, loads a new model
    model_load_run_id = None
    # model_load_run_id = "0b8e182683d34ba4890f0baa4c9af857"  # fw sv mixed
    # model_load_run_id = "a26d0aab8c1b402e8179356e6cf64e3f"  # fw sv mixed continue PG
    # model_load_run_id = "3e842a3acbf648779e82ef22c1dff0ce"  # bidir sv mixed
    model_load_name = 'model'
    # model_load_name = 'epoch-1499'

    run_policy_gradient = True
    run_supervised = True

    description = "Continue PG that was accidentally stopped"

    SUPERVISED_PARAMS = dict(
        save_every=1000,
        epochs=2000,
        lr=0.002,  # default: 0.002
        print_every=1,
        use_cuda=use_cuda,
        test_every=0,
    )

    PG_TRAIN_PARAMS = dict(
        epochs=2000,
        max_actions=5,
        batch_size=5000,
        lr=1e-5,  # default: 0.001
        reward_type='shaped',
        save_every=200,
        forward_only=args.forward_only,
        use_cuda=use_cuda,
        entropy_ratio=0.0,
        test_every=0,
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

    # arc_sampler = arc_task_sampler(
    #     ops,
    #     inputs_small=AUX_PARAMS['small_inputs'],
    #     depth=AUX_PARAMS['depth']
    # )

    arc_sampler = arc_task_sampler2(ops)

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
                               timeout=3000,
                               max_actions=PG_TRAIN_PARAMS['max_actions'],
                               verbose=True)

    SUPERVISED_PARAMS['rollout_fn'] = rollout_fn
    PG_TRAIN_PARAMS['rollout_fn'] = rollout_fn

    if run_supervised:
        supervised_training(net, data, SUPERVISED_PARAMS, AUX_PARAMS,
                            experiment_name='Supervised training 2')

        solved_tasks, attempts_per_task = rollout_fn()
        print(f"solved {len(solved_tasks)} tasks out of {len(attempts_per_task)}")

    if run_policy_gradient:
        policy_gradient(net, policy_gradient_sampler, ops, PG_TRAIN_PARAMS,
                        AUX_PARAMS, experiment_name='Policy gradient 2')

        solved_tasks, attempts_per_task = rollout_fn()
        print(f"solved {len(solved_tasks)} tasks out of {len(attempts_per_task)}")


if __name__ == '__main__':
    arc_training()
