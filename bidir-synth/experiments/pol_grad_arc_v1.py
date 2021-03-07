"""
Code heavily adapted from spinningup:
https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""
import collections
import operator
import random
from typing import List, Callable, Sequence, Dict, Any

import mlflow
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from bidir.task_utils import Task, arc_task
from bidir.utils import save_mlflow_model
from rl.environment import SynthEnv, SynthEnvAction
from rl.ops.operations import Op
import rl.ops.arc_ops
from rl.policy_net import policy_net_arc_v1, PolicyPred


def train(
    task_sampler: Callable[[], Task],
    ops: Sequence[Op],
    policy_net: torch.nn.Module,
    discount_factor: float = 0.99,
    lr: float = 1e-2,
    epochs: int = 50,
    max_actions: int = 100,
    batch_size: int = 5000,
    print_every: int = 1,
    print_rewards_by_task: bool = True,
    reward_type: str = 'shaped',
    save_model: bool = True,
    **kwargs,
):
    env = SynthEnv(task_sampler=task_sampler, ops=ops, max_actions=max_actions)

    def compute_batch_loss(
        batch_preds: List[PolicyPred],
        batch_weights: List[float],
    ):
        batch_logps = []
        for policy_pred in batch_preds:
            op_idx = torch.tensor(policy_pred.op_idx)
            arg_idxs = torch.tensor(policy_pred.arg_idxs)
            op_logits = policy_pred.op_logits
            arg_logits = policy_pred.arg_logits

            op_logp = Categorical(logits=op_logits).log_prob(op_idx)
            arg_logps = Categorical(logits=arg_logits).log_prob(arg_idxs)

            # chain rule: p(action) = p(op) p(arg1 | op) p(arg2 | op) ...
            #             logp(action) = logp(op) + logp(arg1 | op) ...
            logp = op_logp + torch.sum(arg_logps)

            batch_logps.append(logp)

        logps = torch.stack(batch_logps)  # stack to make gradients work
        weights = torch.as_tensor(batch_weights)
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
        batch_preds: List[PolicyPred] = []  # for actions and their logits
        batch_weights: List[float] = []  # R(tau) weighting in policy gradient
        batch_rets: List[float] = []  # for measuring episode returns
        batch_lens: List[int] = []  # for measuring episode lengths

        batch_tasks: List[Task] = []  # for tracking what tasks we train on
        batch_solved: List[bool] = []  # for tracking if we solved the task

        # reset episode-specific variables
        obs = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep
        discount = 1.0

        # collect experience by acting in the environment with current policy
        while True:
            # choose op and arguments
            pred = policy_net(obs.psg)
            act = SynthEnvAction(pred.op_idx, pred.arg_idxs)
            obs, rew, done, _ = env.step(act)

            # save action and logits, reward
            batch_preds.append(pred)
            ep_rews.append(rew * discount)
            discount *= discount_factor

            if done:
                # if len(batch_rets) < 10:
                #     print('Summary: \n' + env.summary())

                # episode is over, record info about episode

                if reward_type == 'shaped':
                    ep_rews = env.episode_rewards()  # type: ignore
                ep_len, ep_ret = len(ep_rews), sum(ep_rews)

                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_tasks.append(env.psg.task)
                batch_solved.append(env.is_solved())

                if reward_type == 'shaped':
                    weights = ep_rews  # = env.episode_rewards()
                elif reward_type == 'to-go':
                    weights = reward_to_go(ep_rews)
                else:
                    weights = [ep_ret] * ep_len

                assert len(weights) == ep_len
                batch_weights += weights

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []
                discount = 1.0

                # end experience loop if we have enough of it
                if len(batch_preds) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss = compute_batch_loss(
            batch_preds=batch_preds,
            batch_weights=batch_weights,
        )
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        return batch_loss, batch_rets, batch_lens, batch_tasks, batch_solved

    try:  # if keyboard interrupt, will save net before exiting!
        # training loop
        for i in range(epochs):
            (batch_loss, batch_rets, batch_lens, batch_tasks,
             batch_solved) = train_one_epoch()
            metrics = dict(epoch=i,
                           loss=float(batch_loss),
                           avg_ret=float(np.mean(batch_rets)),
                           avg_ep_len=float(np.mean(batch_lens)),
                           acc=float(np.mean(batch_solved)))

            mlflow.log_metrics(metrics)

            if metrics["epoch"] % print_every == 0:
                print(
                    'epoch: %3d \t loss: %.3f \t avg_ret: %.3f \t avg_ep_len: %.3f \t acc: %.3f'
                    % (metrics["epoch"], metrics["loss"], metrics["avg_ret"],
                       metrics["avg_ep_len"], metrics["acc"]))

                if print_rewards_by_task:
                    ret_by_task = collections.defaultdict(list)
                    solved_by_task = collections.defaultdict(list)
                    for task, ret, solved in zip(batch_tasks, batch_rets,
                                                 batch_solved):
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
                        print(
                            f"{task}; \t avg_ret={avg_ret:.3f}; \t avg_acc={avg_solved_by_task[task]:.3f}"
                        )

    except KeyboardInterrupt:
        pass

    # save when done, or if we interrupt.
    if save_model:
        save_mlflow_model(policy_net)


def main():
    SEED = 44
    random.seed(44)
    torch.manual_seed(44)

    # See tests/test_on_tasks.py for solutions to tasks
    tasks = [
        # arc_task(task_num=82),
        arc_task(task_num=86),
        # arc_task(task_num=105),
        # arc_task(task_num=115),
        # arc_task(task_num=139),
        # arc_task(task_num=141),
        # arc_task(task_num=149),
        # arc_task(task_num=151),
        # arc_task(task_num=163),
        # arc_task(task_num=178),
        # arc_task(task_num=209),
        # arc_task(task_num=210),
        # arc_task(task_num=240),
        # arc_task(task_num=248),
        # arc_task(task_num=310),
        # arc_task(task_num=379),
    ]

    def task_sampler():
        return random.choice(tasks)
        # return depth_one_random_sample(rl.ops.twenty_four_ops.FORWARD_OPS,
        #                               num_inputs=2,
        #                               max_input_int=24,
        #                               enforce_unique=True).task

    TRAIN_PARAMS = dict(
        discount_factor=0.5,
        epochs=500,
        max_actions=10,
        batch_size=100,
        lr=0.001,
        ops=rl.ops.arc_ops.GRID_OPS,
        reward_type=None,
        print_rewards_by_task=False,
        seed=SEED,
    )
    mlflow.log_params(TRAIN_PARAMS)

    # AUX_PARAMS: Dict[str, Any] = dict(tasks=tasks)
    # mlflow.log_params(AUX_PARAMS)

    policy_net = policy_net_arc_v1(ops=TRAIN_PARAMS["ops"])  # type: ignore

    # model_path = 'models/depth=1_inputs=2_max_input_int=5.pt'
    # policy_net.load_state_dict(unwrap_wrapper_dict(torch.load(model_path)))

    train(
        task_sampler=task_sampler,
        policy_net=policy_net,
        # print_every=10,
        **TRAIN_PARAMS,  # type: ignore
    )


if __name__ == "__main__":
    main()