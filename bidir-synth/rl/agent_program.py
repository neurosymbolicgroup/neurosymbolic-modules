"""
Not sure where best to put this method, so its in a stub file for now.
"""
from typing import Sequence
from rl.ops.operations import Op
from rl.environment import SynthEnvAction, SynthEnv
import rl.agent
from bidir.task_utils import Task
from rl.program import solves


def rl_prog_solves(program: Sequence[SynthEnvAction], task: Task,
                         ops: Sequence[Op]) -> bool:
    env = SynthEnv(
        ops=ops,
        task=task,
        max_actions=len(program),
    )
    agent = rl.agent.ProgrammableAgent(ops, program)

    while not env.done():
        action = agent.choose_action(env.observation())
        env.step(action)
        env.observation().psg.check_invariants()

    if not env.observation().psg.solved():
        return False

    prog = env.observation().psg.get_program()
    assert prog is not None

    return solves(prog, task)
