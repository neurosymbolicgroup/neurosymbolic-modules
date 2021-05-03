"""
Code heavily adapted from spinningup:
https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""
import collections
from collections import Counter
import operator
import random
from typing import List, Callable, Sequence, Dict, Any

import mlflow
import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from bidir.task_utils import Task, twenty_four_task
from bidir.utils import save_mlflow_model, assertEqual
from rl.ops.operations import Op
from rl.random_programs import depth_one_random_24_sample
import rl.ops.twenty_four_ops
from rl.environment import SynthEnv, SynthEnvAction
from rl.policy_net import policy_net_24, PolicyPred


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
    print_rewards_by_task: bool = True,
    # shaped, to-go, otherwise default PG
    reward_type: str = 'shaped',
    save_model: bool = True,
    save_every: int = 500,
    forward_only: bool = False,
    use_cuda: bool = False,
):
    # batch size for running multiple envs at once
    env_batch_size = 20

    envs = [SynthEnv(task_sampler=task_sampler, ops=ops, max_actions=max_actions,
                     forward_only=forward_only)
            for i in range(env_batch_size)]
    # env = SynthEnv(task_sampler=task_sampler, ops=ops, max_actions=max_actions,
    #                  forward_only=forward_only)
    max_arity = policy_net.arg_choice_net.max_arity

    def compute_batch_loss2(
        op_idx_tens: Tensor,
        arg_idx_tens: Tensor,
        op_logits_tens: Tensor,
        arg_logits_tens: Tensor,
        weights: Tensor
    ):
        # not sure if sending to GPU for these calculations are worth it

        N = weights.shape[0]

        assertEqual(op_idx_tens.shape, (N, ))
        assertEqual(arg_idx_tens.shape, (N, max_arity))
        assertEqual(op_logits_tens.shape, (N, len(ops)))
        assertEqual(arg_logits_tens.shape[0:2], (N, max_arity))
        assertEqual(weights.shape, (N, ))

        if use_cuda:
            op_idx_tens = op_idx_tens.cuda()
            arg_idx_tens = arg_idx_tens.cuda()
            op_logits_tens = op_logits_tens.cuda()
            arg_logits_tens = arg_logits_tens.cuda()
            weights = weights.cuda()

        op_logp = Categorical(logits=op_logits_tens).log_prob(op_idx_tens)
        assertEqual(op_logp.shape, (N, ))

        arg_logps = Categorical(logits=arg_logits_tens).log_prob(arg_idx_tens)
        assertEqual(arg_logps.shape, (N, max_arity))

        # sum along the arity axis
        logps = op_logp + torch.sum(arg_logps, dim=1)
        assertEqual(logps.shape, (N, ))

        return -(logps * weights).mean()


    # make optimizer
    optimizer = Adam(policy_net.parameters(), lr=lr)

    def reward_to_go(rews: List[float]) -> List[float]:
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return list(rtgs)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_weights: List[float] = []  # R(tau) weighting in policy gradient
        batch_rets: List[float] = []  # for measuring episode returns
        batch_lens: List[int] = []  # for measuring episode lengths

        # all List[Tensor]
        batch_op_idxs = []  # batches of indices of ops chosen
        batch_arg_idxs = []  # batches of indices of args chosen
        batch_op_logits = []  # batches of op logits
        batch_arg_logits = []  # batches of arg logits

        batch_tasks: List[Task] = []  # for tracking what tasks we train on
        batch_solved: List[bool] = []  # for tracking if we solved the task
        # whether or not ep was solved in single action
        batch_solved_one_step: List[bool] = []

        # reset episode-specific variables
        obss = [env.reset() for env in envs]  # first obs comes from starting distribution
        ep_rews: List[List[float]] = [[] for i in range(len(envs))]  # list for rewards accrued throughout ep
        discounts = [1.0 for i in range(len(envs))]
        envs_op_idxs: List[List[Tensor]] = [[] for i in range(len(envs))]
        envs_arg_idxs: List[List[Tensor]] = [[] for i in range(len(envs))]
        envs_op_logits: List[List[Tensor]] = [[] for i in range(len(envs))]
        envs_arg_logits: List[List[Tensor]] = [[] for i in range(len(envs))]

        num_examples = 0

        # collect experience by acting in the environment with current policy
        batch_done = False
        while not batch_done:
            # choose op and arguments
            preds = policy_net([obs.psg for obs in obss], greedy=False)

            # send to CPU so the large RL batch doesn't fill up GPU
            preds = PolicyPred(preds.op_idxs.cpu(),
                               preds.arg_idxs.cpu(),
                               preds.op_logits.cpu(),
                               preds.arg_logits.cpu())

            for i, env in enumerate(envs):
                act = SynthEnvAction(preds.op_idxs[i].item(),
                                     preds.arg_idxs[i].tolist())
                obss[i], rew, done, _ = env.step(act)

                # save action and logits, reward
                ep_rews[i].append(rew * discounts[i])
                discounts[i] *= discount_factor

                envs_op_idxs[i].append(preds.op_idxs[i])
                envs_arg_idxs[i].append(preds.arg_idxs[i])
                envs_op_logits[i].append(preds.op_logits[i])
                envs_arg_logits[i].append(preds.arg_logits[i])

                if done:
                    batch_op_idxs += envs_op_idxs[i]
                    batch_arg_idxs += envs_arg_idxs[i]
                    batch_op_logits += envs_op_logits[i]
                    batch_arg_logits += envs_arg_logits[i]

                    envs_op_idxs[i] = []
                    envs_arg_idxs[i] = []
                    envs_op_logits[i] = []
                    envs_arg_logits[i] = []

                    if reward_type == 'shaped':
                        ep_rews[i] = env.episode_rewards()  # type: ignore
                    ep_len: int = len(ep_rews[i])
                    ep_ret: float = sum(ep_rews[i])

                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)
                    batch_tasks.append(env.psg.task)
                    batch_solved.append(env.is_solved())
                    batch_solved_one_step.append(env.is_solved() and ep_len == 1)
                    num_examples += ep_len

                    if reward_type == 'shaped':
                        weights = ep_rews[i]  # = env.episode_rewards()
                    elif reward_type == 'to-go':
                        weights = reward_to_go(ep_rews[i])
                    else:
                        weights = [ep_ret] * ep_len

                    assert len(weights) == ep_len
                    batch_weights += weights

                    # reset episode-specific variables
                    obss[i], ep_rews[i], discounts[i] = env.reset(), [], 1.0

                    # end experience loop if we have enough of it
                    if num_examples > batch_size:
                        batch_done = True
                        break

        # put everything into one big batch
        N = num_examples
        op_idx_tens = torch.stack(batch_op_idxs)
        assertEqual(op_idx_tens.shape, (N, ))

        arg_idx_tens = torch.stack(batch_arg_idxs)
        assertEqual(arg_idx_tens.shape, (N, max_arity))

        op_logits_tens = torch.stack(batch_op_logits)
        assertEqual(op_logits_tens.shape, (N, len(ops)))

        arg_logits_tens = torch.stack(batch_arg_logits)
        assertEqual(arg_logits_tens.shape[0:2], (N, max_arity))

        weights = torch.tensor(batch_weights)
        assertEqual(weights.shape, (N, ))

        # take a single policy gradient update step
        batch_loss = compute_batch_loss2(op_idx_tens,
                                         arg_idx_tens,
                                         op_logits_tens,
                                         arg_logits_tens,
                                         weights)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        return (batch_loss, batch_rets, batch_lens, batch_tasks, batch_solved,
                batch_solved_one_step)

    try:  # if keyboard interrupt, will save net before exiting!
        # training loop
        for i in range(epochs):
            (batch_loss, batch_rets, batch_lens, batch_tasks, batch_solved,
             batch_solved_one_step) = train_one_epoch()
            metrics = dict(
                epoch=i,
                loss=float(batch_loss),
                avg_ret=float(np.mean(batch_rets)),
                avg_ep_len=float(np.mean(batch_lens)),
                acc=float(np.mean(batch_solved)),
                one_step=float(np.mean(batch_solved_one_step)),
            )

            mlflow.log_metrics(metrics, step=i)

            if metrics["epoch"] % print_every == 0:
                print(('epoch: %3d \t loss: %.3f \t avg_ret: %.3f \t ' +
                       'avg_ep_len: %.3f \t acc: %.3f \t one_step: %3f') %
                      (metrics["epoch"], metrics["loss"], metrics["avg_ret"],
                       metrics["avg_ep_len"], metrics["acc"],
                       metrics["one_step"]))

                solved_ep_lens = [
                    ep_len
                    for (ep_len, solved) in zip(batch_lens, batch_solved)
                    if solved
                ]
                freq_dict = Counter(solved_ep_lens)
                print(f"solved ep len freqs: {freq_dict}")

                if print_rewards_by_task:
                    print_rewards(batch_tasks, batch_rets, batch_solved)

            if (save_model and save_every > 0 and metrics["epoch"] > 0
                    and metrics["epoch"] % save_every == 0):
                save_mlflow_model(policy_net,
                                  model_name=f"epoch-{metrics['epoch']}")

    except KeyboardInterrupt:
        pass

    # save when done, or if we interrupt.
    if save_model:
        save_mlflow_model(policy_net)


def print_rewards(batch_tasks, batch_rets, batch_solved) -> None:
    ret_by_task = collections.defaultdict(list)
    solved_by_task = collections.defaultdict(list)
    for task, ret, solved in zip(batch_tasks, batch_rets, batch_solved):
        ret_by_task[task].append(ret)
        solved_by_task[task].append(solved)

    avg_ret_by_task = {
        task: np.mean(rets)
        for task, rets in ret_by_task.items()
    }
    avg_solved_by_task = {
        task: np.mean(solves)
        for task, solves in solved_by_task.items()
    }

    for task, avg_ret in sorted(avg_ret_by_task.items(),
                                key=operator.itemgetter(1),
                                reverse=True):
        print(f"{task}; \t avg_ret={avg_ret:.3f};" +
              f"\t avg_acc={avg_solved_by_task[task]:.3f}")


def simon_pol_grad():
    random.seed(44)
    torch.manual_seed(44)

    NUM_OPS = 5
    OPS = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:NUM_OPS]
    MAX_INT = rl.ops.twenty_four_ops.MAX_INT

    def task_sampler():
        return depth_one_random_24_sample(OPS,
                                          num_inputs=2,
                                          max_input_int=10,
                                          enforce_unique=False).task

    load_path = "models/test_save.pt"
    save_path = "models/test_save_pg2.pt"

    TRAIN_PARAMS = dict(
        discount_factor=0.5,
        epochs=500,
        max_actions=10,
        batch_size=1000,
        lr=0.001,
        # ops=OPS,
        reward_type=None,
        print_rewards_by_task=False,
        save_path=save_path,
    )

    AUX_PARAMS: Dict[str, Any] = dict(
        # tasks=tasks,
        # task=TASK,
        model_load_path=load_path,
        num_ops=NUM_OPS,
    )

    mlflow.log_params(TRAIN_PARAMS)
    mlflow.log_params(AUX_PARAMS)

    policy_net = policy_net_24(ops=OPS, max_int=MAX_INT, state_dim=512)

    # policy_net.load_state_dict(torch.load(load_path))
    # policy_net.load_state_dict(unwrap_wrapper_dict(torch.load(load_path)))

    train(
        task_sampler=task_sampler,
        ops=OPS,  # mlflow gives logging error if num_ops > 4
        policy_net=policy_net,
        # print_every=10,
        **TRAIN_PARAMS,  # type: ignore
    )


def main():

    random.seed(44)
    torch.manual_seed(44)

    tasks = [
        twenty_four_task((2, 6, 4), 24),
        twenty_four_task((2, 2), 4),
        twenty_four_task((3, 2), 5),
        twenty_four_task((3, 3), 6),
        twenty_four_task((10, 3), 7),
        twenty_four_task((3, 3), 1),
    ]

    def task_sampler():
        # return random.choice(tasks)
        return depth_one_random_24_sample(rl.ops.twenty_four_ops.FORWARD_OPS,
                                          num_inputs=2,
                                          max_input_int=24,
                                          enforce_unique=True).task

    TRAIN_PARAMS = dict(
        discount_factor=0.5,
        epochs=500,
        max_actions=10,
        batch_size=1000,
        lr=0.001,
        ops=rl.ops.twenty_four_ops.FORWARD_OPS,
        max_int=rl.ops.twenty_four_ops.MAX_INT,
        reward_type=None,
        print_rewards_by_task=False,
    )

    AUX_PARAMS: Dict[str, Any] = dict(tasks=tasks,
                                      # task=TASK,
                                      # model_path=model_path,
                                      )

    mlflow.log_params(TRAIN_PARAMS)
    mlflow.log_params(AUX_PARAMS)

    # task_sampler = lambda: TASK

    policy_net = policy_net_24(
        ops=TRAIN_PARAMS["ops"],  # type: ignore
        max_int=TRAIN_PARAMS["max_int"])  # type: ignore

    # model_path = 'models/depth=1_inputs=2_max_input_int=5.pt'
    # policy_net.load_state_dict(unwrap_wrapper_dict(torch.load(model_path)))

    train(
        task_sampler=task_sampler,
        policy_net=policy_net,
        # print_every=10,
        **TRAIN_PARAMS)  # type: ignore


if __name__ == "__main__":
    main()
