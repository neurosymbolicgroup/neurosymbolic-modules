"""
Code adapted from:
https://github.com/lilianweng/deep-reinforcement-learning-gym/blob/master/playground/policies/dqn.py
"""
from copy import deepcopy
import random
from typing import List, NamedTuple, Sequence, Tuple

import mlflow
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from bidir.task_utils import twenty_four_task
from bidir.utils import save_mlflow_model

from rl.dqn.dqn_models import (SynthQNet, epsilon_greedy_sample,
                               get_relevant_qvals)
from rl.dqn.replay import SynthEnvReplayBuffer, SynthEnvTransition
from rl.ops.operations import Op
import rl.ops.twenty_four_ops
from rl.random_programs import depth_one_random_sample
from rl.environment import SynthEnv
from rl.policy_net import TwentyFourNodeEmbedNet


def perform_action(
    env: SynthEnv,
    sqn: SynthQNet,
    greedy_eps: float,
) -> SynthEnvTransition:
    obs = deepcopy(env.observation())
    act = epsilon_greedy_sample(sqn=sqn, eps=greedy_eps, obs=obs, no_grad=True)

    obs, rew, done, _ = env.step(act)
    obs_next = deepcopy(obs)

    return SynthEnvTransition(
        obs=obs,
        act=act,
        rew=rew,
        obs_next=obs_next,
        done=done,
    )


def train_sqn(
    env: SynthEnv,
    sqn: SynthQNet,
    discount_factor: float,

    # Replay buffer params
    replay_buffer_max_size: int,
    replay_buffer_prefill: int,

    # Train loop config
    num_epochs: int,
    samples_per_epoch: int,  # added to the replay buffer
    opt_steps_per_epoch: int,
    batch_size: int,  # for optimization

    # Train loop config -- epsilon schedule
    eps_init: float,
    eps_final: float,
    eps_decay_per_epoch: float,  # linear decay

    # Train loop config -- optimization details
    internal_loss_scale: float,
    learning_rate: float,

    # Eval params
    eval_every: int,
    eval_greedy_eps: int,
    eval_num_rollouts: int,

    # Etc
    save_model: bool = False,
    **kwargs,
):
    """
    We replace sqn_old with sqn every epoch.
    We perform dqn updates with batch_size samples from the replay memory.
    """
    # Used for DQN fixed-point iteration
    sqn_old = deepcopy(sqn)

    # Only sqn in optimized
    optimizer = Adam(sqn.parameters(), lr=learning_rate)

    # Create replay buffer and prefill with batch_size actions
    assert replay_buffer_prefill <= replay_buffer_max_size
    assert replay_buffer_prefill >= batch_size
    replay_buffer = SynthEnvReplayBuffer(max_size=replay_buffer_max_size)
    while replay_buffer.size < replay_buffer_prefill:
        env.reset()
        while not env.done():
            trans = perform_action(env=env, sqn=sqn, greedy_eps=1.0)
            replay_buffer.add(trans)

    # Main train loop
    try:  # if keyboard interrupt, will save net before exiting!
        cur_eps = eps_init
        for epoch_idx in range(num_epochs):
            # Optimize
            for _ in range(opt_steps_per_epoch):
                trans_batch = replay_buffer.sample(batch_size)

                internal_loss = 0  # consistency between sqn op_qvals and arg_qvals
                external_loss = 0
                for t in trans_batch:
                    q = sqn.forward(obs=t.obs, forced_op_idx=t.act.op_idx)
                    rq: Tensor = get_relevant_qvals(q, arg_idxs=t.act.arg_idxs)

                    q_target = sqn_old.forward(obs=t.obs_next)
                    rq_target: Tensor = get_relevant_qvals(
                        q, arg_idxs=q_target.greedy_arg_idxs)

                    internal_loss += F.smooth_l1_loss(
                        input=rq[1:],
                        target=rq[:-1],
                    ).mean()

                    if t.done:
                        external_loss += F.smooth_l1_loss(
                            input=rq.mean(),
                            target=torch.tensor(t.rew, dtype=torch.float),
                        )
                    else:
                        external_loss += F.smooth_l1_loss(
                            input=rq.mean(),
                            target=t.rew + discount_factor * rq_target.mean(),
                        )

                batch_loss: Tensor = (external_loss + internal_loss_scale *
                                      internal_loss) / batch_size

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # Evaluate
            if eval_every and epoch_idx % eval_every == 0:
                print(f"epoch={epoch_idx}", end="; ")
                print(f"cur_eps={cur_eps:.3f}", end="; ")
                eval_sqn(
                    sqn=sqn,
                    env=env,
                    discount_factor=discount_factor,
                    greedy_eps=eval_greedy_eps,
                    num_rollouts=eval_num_rollouts,
                    print_summary=True,
                )

            # Add new samples to replay buffer
            num_sampled = 0
            while num_sampled < samples_per_epoch:
                env.reset()
                while not env.done():
                    trans = perform_action(env=env,
                                           sqn=sqn,
                                           greedy_eps=cur_eps)
                    replay_buffer.add(trans)
                    num_sampled += 1

            # End of epoch tasks
            sqn_old.load_state_dict(sqn.state_dict())
            cur_eps = max(cur_eps - eps_decay_per_epoch, eps_final)

    except KeyboardInterrupt:
        pass

    if save_model:
        save_mlflow_model(sqn)


def eval_sqn(
    sqn: SynthQNet,
    env: SynthEnv,
    discount_factor: float,
    num_rollouts: int = 50,
    greedy_eps: float = 0,
    print_summary: bool = False,
) -> Tuple[List[float], List[bool]]:
    rewards: List[float] = []
    solves: List[bool] = []
    for _ in range(num_rollouts):
        env.reset()

        rollout_reward = 0
        cur_discount = 1.0
        obs = env.observation()
        while not env.done():
            # Select and perform an action
            act = epsilon_greedy_sample(sqn=sqn, eps=greedy_eps, obs=obs)
            obs, rew, _, _ = env.step(act)

            rollout_reward += cur_discount * rew
            cur_discount *= discount_factor

        rewards.append(rollout_reward)
        solves.append(env.is_solved())

    if print_summary:
        print(f"Eval over {num_rollouts} rollouts", end="; ")
        print(f"avg_rew={np.mean(rewards):.3f}; acc={np.mean(solves):.3f}")

    return rewards, solves


class ExperimentConfig(NamedTuple):
    # Environment config
    ops: Sequence[Op] = rl.ops.twenty_four_ops.FORWARD_OPS
    max_actions: int = 10
    max_int: int = rl.ops.twenty_four_ops.MAX_INT
    discount_factor: float = 0.9

    # Network config
    hidden_dim: int = 256

    # Training config
    replay_buffer_max_size: int = 1000
    replay_buffer_prefill: int = 128

    num_epochs: int = 1000
    samples_per_epoch: int = 10
    opt_steps_per_epoch: int = 32
    batch_size: int = 32

    eps_init: float = 1.0
    eps_final: float = 0.05
    eps_decay_per_epoch: float = 0.001

    learning_rate: float = 0.001
    internal_loss_scale: float = 1.0

    eval_every: int = 1
    save_model: bool = False

    # Eval config
    eval_num_rollouts: int = 50
    eval_greedy_eps: float = 0.05

    # Etc
    seed: int = 42


def main():
    CONFIG = ExperimentConfig()
    TASKS = [
        # twenty_four_task((2, 6, 4), 24),
        # twenty_four_task((2, 2), 4),
        twenty_four_task((3, 2), 5),
        # twenty_four_task((3, 3), 6),
        # twenty_four_task((10, 3), 7),
        # twenty_four_task((3, 3), 1),
    ]

    mlflow.log_params(CONFIG._asdict())

    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)

    def task_sampler():
        return random.choice(TASKS)
        # return depth_one_random_sample(rl.ops.twenty_four_ops.FORWARD_OPS,
        #                               num_inputs=2,
        #                               max_input_int=24,
        #                               enforce_unique=True).task

    env = SynthEnv(
        task_sampler=task_sampler,
        ops=CONFIG.ops,
        max_actions=CONFIG.max_actions,
    )

    sqn = SynthQNet(
        ops=env.ops,
        node_embed_net=TwentyFourNodeEmbedNet(CONFIG.max_int),
        hidden_dim=CONFIG.hidden_dim,
    )

    train_sqn(
        env=env,
        sqn=sqn,
        **CONFIG._asdict(),
    )

    eval_sqn(
        sqn=sqn,
        env=env,
        discount_factor=CONFIG.discount_factor,
        num_rollouts=CONFIG.eval_num_rollouts,
        greedy_eps=CONFIG.eval_greedy_eps,
        print_summary=True,
    )


if __name__ == "__main__":
    main()
