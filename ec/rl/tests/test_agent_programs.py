import unittest
from typing import List, Tuple, Union
from rl.agent import ProgrammableAgent
from rl.create_ops import OP_DICT
from rl.environment import ArcEnvironment
from bidir.task_utils import get_task_examples


class TestProgramAgent(unittest.TestCase):
    def check_program_on_task(
        self,
        task_num: int,
        program: List[Tuple[str, Tuple[int]]],
        train: bool = True,
    ):

        train_exs, test_exs = get_task_examples(task_num, train=train)
        env = ArcEnvironment(train_exs, test_exs, max_actions=-1)
        agent = ProgrammableAgent(OP_DICT, program)

        state = env.state
        while not agent.done():
            action = agent.choose_action(state)
            state, reward, done = env.step(action)

        self.assertTrue(env.done)
        prog = env.state.get_program()
        print(f'Program generated from agent behavior: {prog}')
        self.assertEqual(prog.evaluate(env.state.num_examples),
                         env.state.end.value)

    def get_train_program(
        self,
        task_num: int,
    ) -> Union[str, List[Tuple[str, Tuple[int]]]]:
        if task_num == 56:
            program = [
                ('Black', (0, )),
                ('set_bg', (0, 2)),
                ('crop', (3, )),
                ('1', (0, )),
                ('2', (0, )),
                ('block', (5, 6, 2)),
                ('kronecker', (7, 4)),
                ('unset_bg', (8, 2)),
            ]
            return program
        elif task_num == 86:
            program = [
                ('rotate_cw_inv', (1, )),
                ('rotate_cw_inv', (2, )),
            ]
            return program
        elif task_num == 115:
            program = [
                ('vflip', (0, )),
                ('vstack_pair_cond_inv', (1, 2, None)),
            ]
            return program
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

        print(f"\nSolved {total_solved} ARC tasks with RL programmable agent!")
