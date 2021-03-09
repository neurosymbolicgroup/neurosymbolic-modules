from typing import Dict, Sequence
import torch.nn as nn
import signal
from bidir.task_utils import Task
import random
from rl.environment import SynthEnv, SynthEnvAction
from rl.ops.operations import Op


def policy_rollouts(model: nn.Module, ops: Sequence[Op], tasks: Sequence[Task], timeout: int) -> Dict:
    """
    Timeout is in seconds.
    """
    NUM_ACTIONS = 100

    # divide logits by the temperature before sampling. Higher means more
    # random.
    # between (0, inf]. 0 means argmax, 1 means normal probs, inf means random.
    # TODO: incorporate temperature into policy_net
    # TEMPERATURE = 1

    # each time we do a rollout, just choose a task randomly from those as yet
    # unsolved.
    signal.alarm(timeout)

    solved_tasks = set()
    attempts_per_task = dict(zip(tasks, [0]*len(tasks)))

    try:
        unsolved_tasks = list(tasks)
        while True:
            task = random.choice(unsolved_tasks)
            env = SynthEnv(ops, task, max_actions=NUM_ACTIONS)
            obs = env.reset()

            while not env.done():
                pred = model(obs.psg)
                act = SynthEnvAction(pred.op_idx, pred.arg_idxs)
                obs, rew, done, _ = env.step(act)

            attempts_per_task += obs.action_count_
            if env.is_solved():
                solved_tasks.add(task)

    except Exception as e:
        return solved_tasks, attempts_per_task
