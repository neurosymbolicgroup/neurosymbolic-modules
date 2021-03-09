from typing import Dict, Sequence
import torch.nn as nn
import signal

def policy_rollouts(model: nn.Module, ops: Sequence[Op], tasks: Sequence[Task], timeout: int) -> Dict:
    """
    Timeout is in seconds.
    """
    NUM_ACTIONS = 100
    # divide logits by the temperature before sampling. Higher means more
    # random.
    # between (0, inf]. 0 means argmax, 1 means normal probs, inf means random.
    TEMPERATURE = 1
    # each time we do a rollout, just choose a task randomly from those as yet
    # unsolved.
    signal.alarm(timeout)
    try:
        unsolved_tasks = list(tasks)
        while True:
            task = random.choice(unsolved_tasks)
            env = SynthEnv(task, max_actions=NUM_ACTIONS)
            obs = env.reset()

            while not env.done():
                pred = policy_net(obs.psg)
                obs, rew, done, _ = env.step(pred.action)

    def __init__(
        self,
        ops: Sequence[Op],
        task: Task = None,
        task_sampler: Callable[[], Task] = None,
        max_actions=100,
        solve_reward=100,
        synth_error_penalty=-1,
        timeout_penalty=0,
    ):


    except Exception as e:
        print(type(e))
        assert False
