from typing import Dict, Sequence, Tuple, Set, List
import torch.nn as nn
import time
from bidir.task_utils import Task
import random
from rl.environment import SynthEnv, SynthEnvAction
from rl.ops.operations import Op


def policy_rollouts(models: List[nn.Module], ops: Sequence[Op], tasks: Sequence[Task], timeout: int) -> Tuple[Set[Task], Dict[Task, int]]:
    """
    Timeout is in seconds.
    """
    NUM_ACTIONS = 25

    # divide logits by the temperature before sampling. Higher means more
    # random.
    # between (0, inf]. 0 means argmax, 1 means normal probs, inf means random.
    # TODO: incorporate temperature into policy_net
    # TEMPERATURE = 1

    # each time we do a rollout, just choose a task randomly from those as yet
    # unsolved.

    solved_tasks = set()
    attempts_per_task = dict(zip(tasks, [0]*len(tasks)))
    tries_per_task = dict(zip(tasks, [0]*len(tasks)))
    unsolved_tasks = list(tasks)

    start = time.time()

    while (time.time() - start) < timeout:
        model = random.choice(models)
        task = random.choice(unsolved_tasks)
        tries_per_task[task] += 1

        env = SynthEnv(ops, task, max_actions=NUM_ACTIONS)
        obs = env.reset()

        while not env.done():
            pred = model(obs.psg)
            act = SynthEnvAction(pred.action.op_idx, pred.action.arg_idxs)
            obs, rew, done, _ = env.step(act)

        attempts_per_task[task] += obs.action_count_
        if env.is_solved():
            solved_tasks.add(task)
            print(f"Solved task: {task} in {obs.action_count_} steps after {tries_per_task[task]} rollouts!")
            print(f"Solution: {env.psg.get_program()}")
            unsolved_tasks.remove(task)

    total_rollouts = sum(tries_per_task.values())
    print(f"Attempted {total_rollouts} rollouts in {timeout} seconds")
    print(f"This is equivalent to {sum(attempts_per_task.values())} programs, maybe")
    print(f"Solved {len(solved_tasks)} tasks out of {len(tasks)}")
    print(f"Rollouts per task: {sum(tries_per_task.values()) / len(tasks)}")
    if len(solved_tasks) > 0:
        print(f"Rollouts per task solved: {sum([tries_per_task[t] for t in solved_tasks]) / len(solved_tasks)}")
    if len(solved_tasks) > 0:
        print(f"Attempts per task solved: {sum([attempts_per_task[t] for t in solved_tasks]) / len(solved_tasks)}")
    if len(unsolved_tasks) > 0:
        print(f"Attempts per task unsolved: {sum([attempts_per_task[t] for t in unsolved_tasks]) / len(unsolved_tasks)}")
    if len(unsolved_tasks) > 0:
        print(f"Rollouts per task unsolved: {sum([tries_per_task[t] for t in unsolved_tasks]) / len(unsolved_tasks)}")

    return solved_tasks, attempts_per_task
