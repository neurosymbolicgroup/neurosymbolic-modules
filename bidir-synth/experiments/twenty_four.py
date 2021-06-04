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
from rl.main import policy_gradient, supervised_training, count_parameters, bidir_dataset
from experiments.supervised_training import ActionDataset, program_dataset
import experiments.supervised_training as sv_train
import experiments.policy_gradient as pg
import torch


def training_24():
    print('Twenty four training')
    parser = argparse.ArgumentParser(description='training24')
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument("--forward_only",
                        action="store_true",
                        dest="forward_only")
    parser.add_argument("--dataset_gen",
                        action="store_true",
                        dest="dataset_gen")
    parser.add_argument("--run_supervised",
                        action="store_true",
                        dest="run_supervised")
    parser.add_argument("--run_policy_gradient",
                        action="store_true",
                        dest="run_policy_gradient")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # multithreading doesn't seem to help, so disable
    torch.set_num_threads(1)

    if args.dataset_gen:
        parallel_24_dataset_gen()
        return

    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()

    model_load_run_id = None
    # model_load_run_id = "e0b94b6c83c741b0989679253a57fec3"

    model_load_name = 'model'
    # model_load_name = 'epoch-14000'

    run_supervised = args.run_supervised
    run_policy_gradient = args.run_policy_gradient

    depth = args.depth
    forward_only = args.forward_only
    description = f"Official experiment"

    SUPERVISED_PARAMS = dict(
        save_every=1000,
        epochs=10000,
        lr=0.002,  # default: 0.002
        # print_every=100,
        print_every=1,
        use_cuda=use_cuda,
        test_every=500,
    )

    # TODO: fill in model ID's. this one shown as an example
    if args.run_policy_gradient:
        if args.forward_only:
            if args.seed == 1:
                model_load_run_id = "07a54c05adad4735bc327f4aea072748"
            elif args.seed == 2:
                model_load_run_id = ""
            elif args.seed == 3:
                model_load_run_id = ""
            elif args.seed == 4:
                model_load_run_id = ""
            elif args.seed == 5:
                model_load_run_id = ""
        else: # bidir
            if args.seed == 1:
                model_load_run_id = "07a54c05adad4735bc327f4aea072748"
            elif args.seed == 2:
                model_load_run_id = ""
            elif args.seed == 3:
                model_load_run_id = ""
            elif args.seed == 4:
                model_load_run_id = ""
            elif args.seed == 5:
                model_load_run_id = ""

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
        depth=0,  # depth zero gives the 'mixed' dataset
        # depth=AUX_PARAMS['depth'],
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

    # SUPERVISED_PARAMS['rollout_fn'] = rollout_fn
    # PG_TRAIN_PARAMS['rollout_fn'] = rollout_fn

    if run_supervised:
        supervised_training(net, supervised_data, SUPERVISED_PARAMS, AUX_PARAMS,
                            experiment_name='24 Supervised Training')
    if run_policy_gradient:
        policy_gradient(net, policy_gradient_sampler, ops, PG_TRAIN_PARAMS,
                        AUX_PARAMS, experiment_name='24 Policy Gradient')


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


def parallel_24_dataset_gen():
    args = [(depth, forward_only) for depth in [1, 2, 3, 4]
            for forward_only in [True, False]]

    with Pool() as p:
        p.map(twenty_four_bidir_dataset_gen, args)

    # if unparallel desired
    # for arg in args:
        # arc_bidir_dataset_gen(arg)


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
                  num_samples=10000,
                  forward_only=forward_only,
                  ops=ops)


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


if __name__ == '__main__':
    training_24()
