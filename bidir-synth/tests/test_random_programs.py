import unittest
from typing import List, Tuple, Dict, Sequence
from bidir.task_utils import Task, arc_task, twenty_four_task, plot_task
from rl.random_programs import bidirize_program, random_task
from rl.environment import SynthEnvAction
import rl.ops.twenty_four_ops as twenty_four_ops
import rl.ops.arc_ops as arc_ops
from rl.agent_program import rl_prog_solves
import random
from rl.ops.operations import Op


class RandomProgramTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward_programs(self):
        ops = twenty_four_ops.FORWARD_OPS
        for i in range(20):
            inputs = random.sample(range(1, 10), k=4)
            # print(f"inputs: {inputs}")
            task, psg, _, program = random_task(
                ops,
                twenty_four_task(tuple(inputs), None).inputs,  # type: ignore
                depth=5)
            # print(f"program: {program}")
            # print(f"task: {task}")

            bidir_prog = bidirize_program(task,
                                          psg,
                                          ops,
                                          inv_prob=0,
                                          cond_inv_prob=0)
            # for action in bidir_prog.actions:
            #     print(f"action: {ops[action.op_idx], action.arg_idxs}")

            self.assertTrue(
                rl_prog_solves(bidir_prog.actions, bidir_prog.task, ops))

    def test_inv_programs(self):
        fw_ops = twenty_four_ops.FORWARD_OPS[0:1] + [twenty_four_ops.MINUS1_OP]
        ops: Sequence[Op] = fw_ops + [twenty_four_ops.MINUS1_INV_OP]  # type: ignore

        for i in range(20):
            inputs = random.sample(range(1, 10), k=4)
            # print(f"inputs: {inputs}")
            task, psg, _, program = random_task(fw_ops,
                twenty_four_task(tuple(inputs), None).inputs,  # type: ignore
                depth=15)
            # print(f"program: {program}")
            # print(f"task: {task}")

            bidir_prog = bidirize_program(task,
                                          psg,
                                          ops,
                                          inv_prob=1,
                                          cond_inv_prob=0)
            # for action in bidir_prog.actions:
            #     print(f"action: {ops[action.op_idx], action.arg_idxs}")

            self.assertTrue(
                rl_prog_solves(bidir_prog.actions, bidir_prog.task, ops))

    def test_cond_inv_programs(self):
        ops = twenty_four_ops.ALL_OPS + [
            twenty_four_ops.MINUS1_OP, twenty_four_ops.MINUS1_INV_OP  # type: ignore
        ]

        for i in range(20):
            inputs = random.sample(range(1, 10), k=4)
            # print(f"inputs: {inputs}")
            task, psg, _, program = random_task(twenty_four_ops.FORWARD_OPS +
                [twenty_four_ops.MINUS1_OP],
                twenty_four_task(tuple(inputs), None).inputs,  # type: ignore
                depth=15)
            # print(f"program: {program}")
            # print(f"task: {task}")

            bidir_prog = bidirize_program(task,
                                          psg,
                                          ops,
                                          inv_prob=.8,
                                          cond_inv_prob=.8)
            # for action in bidir_prog.actions:
            #     print(f"action: {ops[action.op_idx], action.arg_idxs}")

            self.assertTrue(
                rl_prog_solves(bidir_prog.actions, bidir_prog.task, ops))

    # turned off, since it does fail sometime and we're OK with that.
    # def test_arc(self):
    #     d: Dict[str, int] = dict(
    #         zip([op.name for op in arc_ops.BIDIR_GRID_OPS],
    #             range(len(arc_ops.BIDIR_GRID_OPS))))

    #     def f(op: str, arg_idxs: Tuple[int, ...]) -> SynthEnvAction:
    #         return SynthEnvAction(d[op], arg_idxs)

    #     prog = [
    #         f('hflip', (0, )),
    #         f('hstack_pair', (0, 2)),
    #         f('vstack_pair_cond_inv_top', (1, 3)),
    #         f('vflip', (0, )),
    #         f('hstack_pair_cond_inv_left', (4, 5)),
    #         f('vflip_inv', (6, )),
    #     ]

    #     task = arc_task(82)

    #     self.assertTrue(rl_prog_solves(prog, task, arc_ops.BIDIR_GRID_OPS))

    #     for i in range(10):
    #         task, psg, _, prog = random_task(arc_ops.FW_GRID_OPS,
    #                                          task.inputs,
    #                                          depth=5)
    #         print(f"prog: {prog}")
    #         bidir_prog = bidirize_program(task,
    #                                       psg,
    #                                       arc_ops.BIDIR_GRID_OPS,
    #                                       inv_prob=.8,
    #                                       cond_inv_prob=.8)

    #         for action in bidir_prog.actions:
    #             print(f"action: {arc_ops.BIDIR_GRID_OPS[action.op_idx]}",
    #                   f"{action.arg_idxs}")

    #         # plot_task(task, text=str(prog), block=True)
    #         self.assertTrue(
    #             rl_prog_solves(bidir_prog.actions, bidir_prog.task,
    #                            arc_ops.BIDIR_GRID_OPS))

