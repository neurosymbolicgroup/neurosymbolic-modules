import unittest
from typing import Union, Tuple, List

from bidir.task_utils import get_arc_task_examples
from rl.agent import ProgrammableAgent, ProgrammbleAgentProgram
from rl.environment import SynthEnv
import rl.ops.arc_ops
import rl.ops.twenty_four_ops


class TestTwentyFourProgramAgent(unittest.TestCase):
    def check_program_on_task(
        self,
        numbers: Tuple[int, int, int, int],
        program: ProgrammbleAgentProgram,
        train: bool = True,
    ):
        train_exs = ((numbers, 24), )
        env = SynthEnv(
            train_exs,
            tuple(),
            rl.ops.twenty_four_ops.ALL_OPS,
            max_actions=len(program),
        )
        agent = ProgrammableAgent(env.ops, program)

        while not env.done():
            action = agent.choose_action(env.observation())
            env.step(action)
            env.observation().psg.check_invariants()

        self.assertTrue(env.observation().psg.solved())

        prog = env.observation().psg.get_program()
        print(
            f'Given input {numbers}, program generated from agent behavior is: {prog}'
        )

        assert prog is not None

        self.assertEqual(prog.evaluate(numbers), 24)

    def get_programs(
        self,
    ) -> List[Tuple[Tuple[int, int, int, int], ProgrammbleAgentProgram]]:
        return [
            (
                # note: if numbers equal in the in values, they get compressed!
                (1, 2, 4, 6),
                [
                    ('sub', (1, 0)),  # 2 - 1 = 1
                    ('mul', (0, 2)),  # 1 * 4 = 4
                    ('mul', (2, 3)),  # 4 * 6 = 24
                ]),
            (
                (1, 3, 5, 7),
                [
                    ('add', (2, 3)),  # 5 + 7 = 12
                    ('mul_cond_inv', (4, 5)),  # 24 = 12 / ?2
                    ('sub_cond_inv1', (6, 1)),  # 2 = 3 - ?1
                ]),
            (
                (104, 2, 6, 4),
                [
                    ('add_cond_inv', (4, 3)),  # 24 = 4 + ?20
                    ('sub_cond_inv2', (5, 2)),  # 20 = ?26 - 6
                    ('div_cond_inv2', (6, 1)),  # 26 = ?52 / 2
                    ('div', (0, 1)),  # 52 = 104 / 2
                ]),
        ]

    def test_on_train_tasks(self):
        total_solved = 0

        for i, (numbers, program) in enumerate(self.get_programs()):
            with self.subTest(i=i):
                self.check_program_on_task(numbers, program)
                total_solved += 1

        print(
            f"\nSolved {total_solved} 24 game tasks with RL programmable agent."
        )


class TestArcProgramAgent(unittest.TestCase):
    def check_program_on_task(
        self,
        task_num: int,
        program: ProgrammbleAgentProgram,
        train: bool = True,
    ):
        train_exs, test_exs = get_arc_task_examples(task_num, train=train)
        env = SynthEnv(
            tuple(((i, ), o) for i, o in train_exs),
            tuple(((i, ), o) for i, o in test_exs),
            rl.ops.arc_ops.ALL_OPS,
            max_actions=len(program),
        )
        agent = ProgrammableAgent(env.ops, program)

        while not env.done():
            action = agent.choose_action(env.observation())
            env.step(action)
            env.observation().psg.check_invariants()

        self.assertTrue(env.observation().psg.solved())

        prog = env.observation().psg.get_program()
        print(f'Task {task_num} program generated from agent behavior: {prog}')

        for (in_grid, out_grid) in train_exs + test_exs:
            assert prog is not None
            self.assertEqual(prog.evaluate((in_grid, )),
                             out_grid)  # type: ignore

    def get_train_program(
        self,
        task_num: int,
    ) -> Union[str, ProgrammbleAgentProgram]:
        # yapf: disable
        if task_num == 56:
            return [
                ('Color.BLACK', (0, )),
                ('set_bg', (0, 2)),
                ('crop', (3, )),
                ('1', (0, )),
                ('2', (0, )),
                ('block', (5, 6, 2)),
                ('kronecker', (7, 4)),
                ('unset_bg', (8, 2)),
            ]
        elif task_num == 82:
            return [('hflip', (0, )),
                    ('hstack_pair', (0, 2)),
                    ('vstack_pair_cond_inv_top', (1, 3)),
                    ('vflip', (0, )),
                    ('hstack_pair_cond_inv_left', (4, 5)),
                    ('vflip_inv', (6, ))]
        elif task_num == 86:
            return [
                ('rotate_cw_inv', (1, )),
                ('rotate_cw_inv', (2, )),
            ]
        elif task_num == 115:
            return [
                ('vflip', (0, )),
                ('vstack_pair_cond_inv_top', (1, 2)),
            ]
        else:
            return "No program"
        # yapf: enable

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
