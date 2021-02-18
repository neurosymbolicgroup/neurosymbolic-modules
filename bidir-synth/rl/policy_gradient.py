"""
Code heavily adapted from spinningup:
https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""
import random
from typing import List, Callable, Sequence, Dict, Any

import mlflow
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from rl.ops.operations import Op
from rl.random_programs import random_program, depth_one_random_sample
import rl.ops.twenty_four_ops
from rl.environment import SynthEnv, SynthEnvAction
from rl.policy_net import policy_net_24, PolicyPred
from rl.supervised_training import unwrap_wrapper_dict
from bidir.task_utils import Task, twenty_four_task


def train(
    task_sampler: Callable[[], Task],
    ops: Sequence[Op],
    policy_net: torch.nn.Module,
    max_int: int,
    discount_factor: float = 0.99,
    lr: float = 1e-2,
    epochs: int = 50,
    max_actions: int = 100,
    batch_size: int = 5000,
    print_every: int = 1,
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
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += reward_to_go(ep_rews)
                # batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []
                discount = 1.0

                # end experience loop if we have enough of it
                if len(batch_preds) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_batch_loss(
            batch_preds=batch_preds,
            batch_weights=batch_weights,
        )
        batch_loss.backward()
        optimizer.step()

        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        metrics = dict(
            epoch=i,
            loss=float(batch_loss),
            avg_ret=float(np.mean(batch_rets)),
            avg_ep_len=float(np.mean(batch_lens)),
        )

        if metrics["epoch"] % print_every == 0:
            print(
                'epoch: %3d \t loss: %.3f \t avg_ret: %.3f \t avg_ep_len: %.3f'
                % (metrics["epoch"], metrics["loss"], metrics["avg_ret"],
                   metrics["avg_ep_len"]))

        mlflow.log_metrics(metrics)


def main():
    random.seed(42)
    torch.manual_seed(42)

    # def task_sampler():
    #     return depth_one_random_sample(rl.ops.twenty_four_ops.FORWARD_OPS,
    #                                    num_inputs=2,
    #                                    max_input_int=5,
    #                                    enforce_unique=True)

    TASK = twenty_four_task((8, 8), 16)
    task_sampler = lambda: TASK

    policy_net = policy_net_24(rl.ops.twenty_four_ops.FORWARD_OPS,
                               max_int=rl.ops.twenty_four_ops.MAX_INT)

    model_path = 'models/depth=1_inputs=2_max_input_int=5.pt'
    policy_net.load_state_dict(unwrap_wrapper_dict(torch.load(model_path)))


    TRAIN_PARAMS = dict(
        discount_factor=0.9,
        epochs=500,
        max_actions=10,
        batch_size=1000,
        lr=0.01,
        ops=rl.ops.twenty_four_ops.FORWARD_OPS,
        max_int=rl.ops.twenty_four_ops.MAX_INT,
    )

    AUX_PARAMS: Dict[str, Any] = dict(
        task=TASK,
        # model_path=model_path,
    )

    mlflow.log_params(TRAIN_PARAMS)
    mlflow.log_params(AUX_PARAMS)

    train(task_sampler=task_sampler,
          policy_net=policy_net,
          # print_every=10,
          **TRAIN_PARAMS)  # type: ignore


if __name__ == "__main__":
    main()
