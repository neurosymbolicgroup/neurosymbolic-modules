import unittest

from dreamcoder.domains.arc.bidir.task_utils import get_task_grid_pairs
import dreamcoder.domains.arc.bidir.primitives.functions as F


class TestOnTasks(unittest.TestCase):
    def check_arc_task(self, task_num, program, train=True):
        grid_pairs = get_task_grid_pairs(task_num, train=train)
        for in_grid, out_grid in grid_pairs:
            pred_grid = program(in_grid)

            self.assertEqual(
                pred_grid,
                out_grid,
                msg=(f"Didn't solve ARC task #{task_num}\n"
                     f"in  : {in_grid}\n"
                     f"out : {out_grid}\n"
                     f"pred: {pred_grid}\n"),
            )

    def test_train_0(self):
        self.check_arc_task(
            task_num=0,
            program=lambda x: F.kronecker_(F.color_i_to_j_(x)(0)(-1))(x),
        )

    def test_train_222(self):
        self.check_arc_task(
            task_num=222,
            program=lambda x: F.inflate_(x)(3),
        )
