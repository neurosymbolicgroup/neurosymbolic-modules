"""
Code heavily adapted from spinningup:
https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py
"""

from copy import deepcopy
import random
from typing import List

import mlflow
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import rl.ops.twenty_four_ops
from rl.environment import SynthEnv, SynthEnvAction, SynthEnvObservation
from rl.policy_net import OpNet24


def train(
    train_exs,
    test_exs,
    ops,
    max_int,
    discount_factor=0.99,
    lr=1e-2,
    epochs=50,
    max_actions=100,
    batch_size=5000,
):
    # TODO: Make environment sample from different train_exs and test_exs
    env = SynthEnv(train_exs, test_exs, ops, max_actions=max_actions)

    opNet = OpNet24(ops, max_int=max_int)

    # make function to compute op distribution
    def get_op_policy(obs: SynthEnvObservation):
        logits = opNet(obs.psg)
        return Categorical(logits=logits)

    # make action op selection function
    def get_op_idx(obs: SynthEnvObservation):
        return get_op_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_batch_loss(
        batch_obs: List[SynthEnvObservation],
        batch_acts: List[SynthEnvAction],
        batch_weights: List[float],
    ):
        batch_logps = []
        for obs, act in zip(batch_obs, batch_acts):
            op_idx = torch.tensor(act.op_idx)
            batch_logps.append(get_op_policy(obs).log_prob(op_idx))

        logps = torch.stack(batch_logps)  # stack to make gradients work
        weights = torch.as_tensor(batch_weights)
        return -(logps * weights).mean()

    # make optimizer
    optimizer = Adam(opNet.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs: List[SynthEnvObservation] = []  # for observations
        batch_acts: List[SynthEnvAction] = []  # for actions
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
            # save obs
            # TODO: Make observation easier to copy.
            batch_obs.append(deepcopy(obs))

            # Choose op
            op_idx = get_op_idx(obs)
            op = env.ops[op_idx]

            # TODO: Use neural net to choose args
            # Choose args randomly from now
            # Currently only choosing from grounded nodes
            # TODO: Choose from more than just grounded nodes
            grounded_nodes = [
                v for v in obs.psg.get_value_nodes() if obs.psg.is_grounded(v)
            ]
            arg_nodes = tuple(
                random.choice(grounded_nodes) for _ in op.arg_types)

            act = SynthEnvAction(op_idx, arg_nodes)
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew * discount)
            discount *= discount_factor

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_batch_loss(
            batch_obs=batch_obs,
            batch_acts=batch_acts,
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

        print('epoch: %3d \t loss: %.3f \t avg_ret: %.3f \t avg_ep_len: %.3f' %
              (metrics["epoch"], metrics["loss"], metrics["avg_ret"],
               metrics["avg_ep_len"]))

        mlflow.log_metrics(metrics)


def main():
    random.seed(42)
    torch.manual_seed(42)

    TRAIN_EXS = (((1, 2, 3, 4), 24), )
    TRAIN_PARAMS = dict(
        discount_factor=0.99,
        epochs=100,
        max_actions=100,
        batch_size=1000,
    )

    mlflow.log_params(TRAIN_PARAMS)

    # TODO: Use more than just forward_ops
    train(
        train_exs=TRAIN_EXS,
        test_exs=tuple(),
        ops=rl.ops.twenty_four_ops.FORWARD_OPS,
        max_int=rl.ops.twenty_four_ops.MAX_INT,
        **TRAIN_PARAMS,
    )


if __name__ == "__main__":
    main()
