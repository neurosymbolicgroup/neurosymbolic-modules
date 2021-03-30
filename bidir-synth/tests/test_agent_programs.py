import unittest
from typing import Tuple, List

from rl.environment import SynthEnvAction
import rl.ops.arc_ops
import rl.ops.twenty_four_ops
from bidir.task_utils import twenty_four_task, arc_task, Task
from rl.agent_program import rl_prog_solves


class TestAgentPrograms(unittest.TestCase):
    def twenty_four_tasks_and_programs(
            self) -> List[Tuple[Task, List[SynthEnvAction]]]:

        d = dict(
            zip(rl.ops.twenty_four_ops.OP_DICT.keys(),
                range(len(rl.ops.twenty_four_ops.ALL_OPS))))

        def f(op: str, arg_idxs: Tuple[int, ...]) -> SynthEnvAction:
            return SynthEnvAction(d[op], arg_idxs)

        # note: if numbers equal in the in values, they get compressed!
        return [
            (
                twenty_four_task((1, 2, 4, 6), 24),
                [
                    f('sub', (1, 0)),  # 2 - 1 = 1
                    f('mul', (0, 2)),  # 1 * 4 = 4
                    f('mul', (2, 3)),  # 4 * 6 = 24
                ]),
            (
                twenty_four_task((1, 3, 5, 7), 24),
                [
                    f('add', (2, 3)),  # 5 + 7 = 12
                    f('mul_cond_inv1', (4, 5)),  # 24 = 12 / ?2
                    f('sub_cond_inv1', (6, 1)),  # 2 = 3 - ?1
                ]),
            (
                twenty_four_task((104, 2, 6, 4), 24),
                [
                    f('add_cond_inv1', (4, 3)),  # 24 = 4 + ?20
                    f('sub_cond_inv2', (5, 2)),  # 20 = ?26 - 6
                    f('div_cond_inv2', (6, 1)),  # 26 = ?52 / 2
                    f('div', (0, 1)),  # 52 = 104 / 2
                ]),
        ]

    def test_twenty_four_programs(self):
        total_solved = 0

        for i, (task,
                program) in enumerate(self.twenty_four_tasks_and_programs()):
            with self.subTest(i=i):
                self.assertTrue(
                    rl_prog_solves(program, task,
                                   rl.ops.twenty_four_ops.ALL_OPS))
                total_solved += 1

        print(
            f"\nSolved {total_solved} 24 game tasks with RL programmable agent."
        )

    def arc_tasks_and_programs(
            self) -> List[Tuple[Task, List[SynthEnvAction]]]:

        d = dict(
            zip(rl.ops.arc_ops.OP_DICT.keys(),
                range(len(rl.ops.arc_ops.ALL_OPS))))

        def f(op: str, arg_idxs: Tuple[int, ...]) -> SynthEnvAction:
            return SynthEnvAction(d[op], arg_idxs)

        return [
            (arc_task(56), [
                f('Color.BLACK', (0, )),
                f('set_bg', (0, 2)),
                f('crop', (3, )),
                f('1', (0, )),
                f('2', (0, )),
                f('block', (5, 6, 2)),
                f('kronecker', (7, 4)),
                f('unset_bg', (8, 2)),
            ]),
            (arc_task(82), [
                f('hflip', (0, )),
                f('hstack_pair', (0, 2)),
                f('vstack_pair_cond_inv_top', (1, 3)),
                f('vflip', (0, )),
                f('hstack_pair_cond_inv_left', (4, 5)),
                f('vflip_inv', (6, )),
            ]),
            (arc_task(86), [
                f('rotate_cw_inv', (1, )),
                f('rotate_cw_inv', (2, )),
            ]),
            (arc_task(115), [
                f('vflip', (0, )),
                f('vstack_pair_cond_inv_top', (1, 2)),
            ]),
            (arc_task(288), [
                f('Color.BLACK', (0, )),
                f('set_bg', (0, 2)),
                f('colors', (3, )),
                f('length', (4, )),
                f('inflate', (0, 5)),
            ])
        ]

    def test_arc_programs(self):
        total_solved = 0

        for i, (task, program) in enumerate(self.arc_tasks_and_programs()):
            with self.subTest(i=i):
                self.assertTrue(
                    rl_prog_solves(program, task,
                                   rl.ops.arc_ops.ALL_OPS))
                total_solved += 1

        print(f"\nSolved {total_solved} ARC tasks with RL programmable agent.")
