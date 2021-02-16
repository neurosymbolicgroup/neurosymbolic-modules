"""
Code heavily adapted from spinningup:
https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""
import random
from typing import List

import mlflow
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import rl.ops.twenty_four_ops
from rl.environment import SynthEnv, SynthEnvAction
from rl.policy_net import policy_net_24, PolicyPred
from bidir.task_utils import Task


def train(
    task,
    ops,
    max_int,
    discount_factor=0.99,
    lr=1e-2,
    epochs=50,
    max_actions=100,
    batch_size=5000,
    print_every=1,
):
    env = SynthEnv(task, ops, max_actions=max_actions)

    policy_net = policy_net_24(ops, max_int=max_int)

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

            # note: we're assuming that the policy net chose the op and args
            # via a categorial distribution
            # we could change PolicyPred to just return the log_prob of the op
            # and args it chose, but this allows us to train via other methods
            # if desired
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

    TASK = Task(((8, ), (8, )), (24, ))
    TRAIN_PARAMS = dict(
        task=TASK,
        discount_factor=0.9,
        epochs=500,
        max_actions=20,
        batch_size=1000,
        lr=0.01,
        # print_every=10,
        ops=rl.ops.twenty_four_ops.FORWARD_OPS,
        max_int=rl.ops.twenty_four_ops.MAX_INT,
    )

    mlflow.log_params(TRAIN_PARAMS)

    # TODO: Use more than just forward_ops
    train(**TRAIN_PARAMS, )


if __name__ == "__main__":
    main()
