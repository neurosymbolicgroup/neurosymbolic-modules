import unittest
from typing import Union

from bidir.task_utils import get_task_examples
from rl.agent import ProgrammableAgent, ProgrammbleAgentProgram
from rl.create_ops import OP_DICT
from rl.environment import SynthEnv


class TestTwentyFourProgramAgnet(unittest.TestCase):
    def check_program_on_task(
        self,
        task_num: int,
        program: ProgrammbleAgentProgram,
    ):
        train_exs, test_exs = get_task_examples(task_num, train=train)
        env = SynthEnv(train_exs, test_exs, max_actions=len(program))
        agent = ProgrammableAgent(OP_DICT, program)

        while not env.done:
            action = agent.choose_action(env.observation)
            env.step(action)
            env.observation.psg.check_invariants()

        self.assertTrue(env.observation.psg.solved())

        prog = env.observation.psg.get_program()
        print(f'Program generated from agent behavior: {prog}')

        for (in_grid, out_grid) in train_exs + test_exs:
            self.assertEqual(prog.evaluate(in_grid), out_grid)  # type: ignore

class TestArcProgramAgent(unittest.TestCase):
    def check_program_on_task(
        self,
        task_num: int,
        program: ProgrammbleAgentProgram,
        train: bool = True,
    ):
        train_exs, test_exs = get_task_examples(task_num, train=train)
        env = SynthEnv(train_exs, test_exs, max_actions=len(program))
        agent = ProgrammableAgent(OP_DICT, program)

        while not env.done:
            action = agent.choose_action(env.observation)
            env.step(action)
            env.observation.psg.check_invariants()

        self.assertTrue(env.observation.psg.solved())

        prog = env.observation.psg.get_program()
        print(f'Program generated from agent behavior: {prog}')

        for (in_grid, out_grid) in train_exs + test_exs:
            self.assertEqual(prog.evaluate(in_grid), out_grid)  # type: ignore

    def get_train_program(
        self,
        task_num: int,
    ) -> Union[str, ProgrammbleAgentProgram]:
        if task_num == 56:
            return [
                ('Black', (0, )),
                ('set_bg', (0, 2)),
                ('crop', (3, )),
                ('1', (0, )),
                ('2', (0, )),
                ('block', (5, 6, 2)),
                ('kronecker', (7, 4)),
                ('unset_bg', (8, 2)),
            ]
        elif task_num == 82:
            return [
                ('hflip', (0, )),
                ('hstack_pair', (0, 2)),
                ('vstack_pair_cond_inv', (1, 3, None)),
                ('vflip', (0, )),
                ('hstack_pair_cond_inv', (4, 5, None)),
                ('vflip_inv', (6, ))
            ]
        elif task_num == 86:
            return [
                ('rotate_cw_inv', (1, )),
                ('rotate_cw_inv', (2, )),
            ]
        elif task_num == 115:
            return [
                ('vflip', (0, )),
                ('vstack_pair_cond_inv', (1, 2, None)),
            ]
        else:
            return "No program"

    def test_on_train_tasks(self):
        total_solved = 0

        for task_num in range(400):
            program = self.get_train_program(task_num)
            if isinstance(program, str):
                continue

            with self.subTest(task_num=task_num):
                self.assertNotEqual(program, None,
                                    (f"program for {task_num} is None. "
                                     f"Did you forget to 'return program'?"))
                self.check_program_on_task(task_num, program)
                total_solved += 1

        print(f"\nSolved {total_solved} ARC tasks with RL programmable agent.")
