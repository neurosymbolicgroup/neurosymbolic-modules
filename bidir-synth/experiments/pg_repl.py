"""
Code heavily adapted from spinningup:
https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""
from collections import Counter
from typing import List, Callable, Sequence, Dict, Set

import time
import math
import mlflow
import numpy as np
import torch
from torch import Tensor
import torch.optim

from bidir.task_utils import Task
from bidir.utils import assertEqual
import bidir.utils as utils
from rl.ops.operations import Op
from rl.environment import SynthEnv, SynthEnvAction


def train(
    task_sampler: Callable[[], Task],
    ops: Sequence[Op],
    policy_net: torch.nn.Module,
    discount_factor: float = 0.99,
    lr: float = 1e-2,
    epochs: int = 50,
    max_actions: int = 100,
    # batch size for a loss update
    batch_size: int = 5000,
    print_every: int = 1,
    print_rewards_by_task: bool = False,
    # shaped, to-go, otherwise default PG
    reward_type: str = 'shaped',
    save_model: bool = True,
    save_every: int = 500,
    forward_only: bool = False,
    use_cuda: bool = False,
):
    # batch size for running multiple envs at once
    # what's the best size? might be dependent on resources being used
    # I've found that smaller batch sizes are faster for some reason... setting
    # to 1 is the fastest. wth :(
    env_batch_size = 1
    print('warning: small env batch size')

    # sampler = utils.repeat_n_times(task_sampler, n=16)
    sampler = task_sampler

    envs = [SynthEnv(task_sampler=sampler, ops=ops, max_actions=max_actions,
                     forward_only=forward_only)
            for i in range(env_batch_size)]
    max_arity = policy_net.arg_choice_net.max_arity

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    task_to_solved: Dict[Task, bool] = {}
    task_to_attempts: Dict[Task, Set[SynthEnvAction]] = {}
    solved_ops: Dict[int, int] = {i: 0 for i in range(len(ops))}
    ops_tried: Dict[int, int] = {i: 0 for i in range(len(ops))}

    if use_cuda:
        policy_net = policy_net.cuda()
        criterion = criterion.cuda()

    def compute_batch_loss(
        op_idx_tens: Tensor,
        arg_idx_tens: Tensor,
        op_logits_tens: Tensor,
        arg_logits_tens: Tensor
    ):

        """
        Does the loss from the REPL paper: if you solve the task, train to
        predict it.
        """
        N = op_idx_tens.shape[0]
        max_nodes = policy_net.max_nodes

        assertEqual(arg_idx_tens.shape, (N, max_arity))
        assertEqual(op_logits_tens.shape, (N, len(ops)))
        assertEqual(arg_logits_tens.shape, (N, max_arity, max_nodes))

        arg_logits_tens2 = arg_logits_tens.permute(0, 2, 1)
        assertEqual(arg_logits_tens2.shape, (N, max_nodes, max_arity))

        if use_cuda:
            op_idx_tens = op_idx_tens.cuda()
            arg_idx_tens = arg_idx_tens.cuda()
            op_logits_tens = op_logits_tens.cuda()
            arg_logits_tens2 = arg_logits_tens2.cuda()

        op_loss = criterion(op_logits_tens, op_idx_tens)
        arg_loss = criterion(arg_logits_tens2, arg_idx_tens)
        combined_loss = op_loss + arg_loss
        return combined_loss

    def gradient_update(batch_op_idxs, batch_arg_idxs, batch_op_logits,
                        batch_arg_logits):
        # put everything into one big batch
        op_idx_tens = torch.stack(batch_op_idxs)
        N = op_idx_tens.shape[0]

        arg_idx_tens = torch.stack(batch_arg_idxs)
        assertEqual(arg_idx_tens.shape, (N, max_arity))

        op_logits_tens = torch.stack(batch_op_logits)
        assertEqual(op_logits_tens.shape, (N, len(ops)))

        arg_logits_tens = torch.stack(batch_arg_logits)
        assertEqual(arg_logits_tens.shape[0:2], (N, max_arity))

        # take a single policy gradient update step
        batch_loss = compute_batch_loss(op_idx_tens,
                                        arg_idx_tens,
                                        op_logits_tens,
                                        arg_logits_tens)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # for training policy
    def train_one_epoch():
        # number of actions from solved tasks to train on
        grad_batch_size = 128

        # all List[Tensor]
        batch_op_idxs = []  # batches of indices of ops chosen
        batch_arg_idxs = []  # batches of indices of args chosen
        batch_op_logits = []  # batches of op logits
        batch_arg_logits = []  # batches of arg logits

        epoch_lens: List[int] = []  # for measuring episode lengths
        epoch_tasks: List[Task] = []  # for tracking what tasks we train on
        epoch_solved: List[bool] = []  # for tracking if we solved the task
        # whether or not ep was solved in single action
        epoch_solved_one_step: List[bool] = []

        # reset episode-specific variables
        # print('warning: not resetting')  # should do env.reset() not env.obs
        obss = [env.reset() for env in envs]  # first obs comes from starting distribution
        envs_op_idxs: List[List[Tensor]] = [[] for i in range(len(envs))]
        envs_arg_idxs: List[List[Tensor]] = [[] for i in range(len(envs))]
        envs_op_logits: List[List[Tensor]] = [[] for i in range(len(envs))]
        envs_arg_logits: List[List[Tensor]] = [[] for i in range(len(envs))]

        num_examples = 0

        # collect experience by acting in the environment with current policy
        batch_done = False
        while len(epoch_lens) < batch_size:
            # choose op and arguments
            preds = policy_net([obs.psg for obs in obss], greedy=False)

            for i, env in enumerate(envs):
                act = SynthEnvAction(preds.op_idxs[i].item(),
                                     preds.arg_idxs[i].tolist())
                ops_tried[act.op_idx] += 1
                obss[i], rew, done, _ = env.step(act)

                envs_op_idxs[i].append(preds.op_idxs[i])
                envs_arg_idxs[i].append(preds.arg_idxs[i])
                envs_op_logits[i].append(preds.op_logits[i])
                envs_arg_logits[i].append(preds.arg_logits[i])

                if done:
                    ep_len: int = len(env.actions_applied)
                    solved = env.is_solved()
                    epoch_lens.append(ep_len)
                    if len(epoch_lens) == batch_size:
                        print(f'last task: {env.task}')

                    epoch_tasks.append(env.task)
                    epoch_solved.append(solved)
                    epoch_solved_one_step.append(solved and ep_len == 1)

                    if env.is_solved():
                        task_to_solved[env.task] = (True, act.op_idx)
                        solved_ops[act.op_idx] += 1
                        # only add actions that were used in the final solution
                        action_steps = env.psg.actions_in_program()
                        assert action_steps is not None  # for type checking..

                        batch_op_idxs += [envs_op_idxs[i][j]
                                          for j in action_steps]
                        batch_arg_idxs += [envs_arg_idxs[i][j]
                                           for j in action_steps]
                        batch_op_logits += [envs_op_logits[i][j]
                                            for j in action_steps]
                        batch_arg_logits += [envs_arg_logits[i][j]
                                             for j in action_steps]
                        num_examples += len(action_steps)
                        if num_examples > grad_batch_size:
                            # print('update')
                            gradient_update(batch_op_idxs,
                                            batch_arg_idxs,
                                            batch_op_logits,
                                            batch_arg_logits)
                            batch_op_idxs = []
                            batch_arg_idxs = []
                            batch_op_logits = []
                            batch_arg_logits = []
                            num_examples = 0
                    else:
                        if env.task not in task_to_solved:
                            task_to_solved[env.task] = (False, 0)

                    obss[i] = env.reset()
                    envs_op_idxs[i] = []
                    envs_arg_idxs[i] = []
                    envs_op_logits[i] = []
                    envs_arg_logits[i] = []

        if len(batch_op_idxs) > 0:
            print('update')
            gradient_update(batch_op_idxs,
                            batch_arg_idxs,
                            batch_op_logits,
                            batch_arg_logits)

        return (epoch_lens, epoch_tasks, epoch_solved,
                epoch_solved_one_step)

    try:  # if keyboard interrupt, will save net before exiting!
        # training loop
        for epoch in range(epochs):
            start = time.time()

            (epoch_lens, epoch_tasks, epoch_solved,
             epoch_solved_one_step) = train_one_epoch()

            duration = time.time() - start
            m = math.floor(duration / 60)
            s = duration - m * 60
            duration_str = f'{m}m {s:.1f}s'

            metrics = dict(
                epoch=epoch,
                avg_ep_len=float(np.mean(epoch_lens)),
                acc=float(np.mean(epoch_solved)),
                one_step=float(np.mean(epoch_solved_one_step)),
            )

            mlflow.log_metrics(metrics, step=epoch)

            if metrics["epoch"] % print_every == 0:
                print(f"Epoch {epoch} completed ({duration_str})",
                      f"\tavg_ep_len: {metrics['avg_ep_len']:.3f}",
                      f"\tacc: {metrics['acc']:.5f}",
                      f"\tone_step: {metrics['one_step']:.3f}",
                      f"\tsolved: {sum([t[0] for t in task_to_solved.values()])}")

                solved_ep_lens = [
                    ep_len
                    for (ep_len, solved) in zip(epoch_lens, epoch_solved)
                    if solved
                ]
                freq_dict = Counter(solved_ep_lens)
                print(f"solved ep len freqs: {freq_dict}")

            if epoch % 1 == 0:
                # for task, solved in task_to_solved.items():
                #     if solved[0]:
                #         print(f'({solved}) {task}')
                # for task, solved in task_to_solved.items():
                #     if not solved[0]:
                #         print(f'({solved}) {task}')
                print('ops:')
                for op, tried in ops_tried.items():
                    print(f"{op}: tried: {tried}, in solved: {solved_ops[op]}")

            if (save_model and save_every > 0 and metrics["epoch"] > 0
                    and metrics["epoch"] % save_every == 0):
                utils.save_mlflow_model(policy_net,
                                        model_name=f"epoch-{metrics['epoch']}")

    except KeyboardInterrupt:
        pass

    # save when done, or if we interrupt.
    if save_model:
        utils.save_mlflow_model(policy_net)
