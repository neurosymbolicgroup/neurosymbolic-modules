from typing import Dict, Sequence, Tuple, Set, List
import torch.nn as nn
import time
from bidir.task_utils import Task
import random
from rl.environment import SynthEnv, SynthEnvAction
from rl.ops.operations import Op

def policy_rollouts(model: nn.Module,
                    ops: Sequence[Op],
                    tasks: Sequence[Task],
                    timeout: int,
                    max_actions: int = 25,
                    verbose: bool = True) -> Tuple[Set[Task], Dict[Task, int]]:
    """
    Timeout is in seconds.
    """
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

    while unsolved_tasks and (time.time() - start) < timeout:
        batch_size = 1
        tasks_batch = random.sample(unsolved_tasks, k=batch_size)
        for task in tasks_batch:
            tries_per_task[task] += 1

        envs = [SynthEnv(ops, task, max_actions=max_actions)
                for task in tasks_batch]
        obss = [env.reset() for env in envs]

        for i in range(max_actions):
            # TODO: don't need forward pass for the solved envs
            preds = model([obs.psg for obs in obss], greedy=False)

            for i, env in enumerate(envs):
                if not env.is_solved():
                    act = SynthEnvAction(preds.op_idxs[i].item(),
                                         preds.arg_idxs[i].tolist())
                    obss[i], rew, done, _ = env.step(act)

                    if env.is_solved():
                        task = tasks_batch[i]
                        solved_tasks.add(task)
                        if verbose:
                            print(f"Solved task: {task} in {obss[i].action_count_} steps after {tries_per_task[task]} rollouts!")
                            print(f"Solution: {env.psg.get_program()}")
                        unsolved_tasks.remove(task)

        # for env in envs:
        #     if not env.is_solved():
        #         print(f"task: {env.task.target[0]}")
        #         print(env.actions_applied)

        # for i, task in enumerate(tasks_batch):
        for task in tasks_batch:
            # attempts_per_task[task] += obss[i].action_count_
            attempts_per_task[task] += 1

    total_rollouts = sum(tries_per_task.values())
    if verbose:
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
