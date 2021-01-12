import unittest
import numpy as np

from dreamcoder.domains.arc.bidir.task_utils import get_task_grid_pairs
import dreamcoder.domains.arc.bidir.primitives.functions as F


class TestOnTasks(unittest.TestCase):
    def check_arc_train_task(self, task_num, program):
        grid_pairs = get_task_grid_pairs(task_num, train=True)
        for in_grid, out_grid in grid_pairs:
            pred_grid = program(in_grid)

            self.assertEqual(
                pred_grid,
                out_grid,
                msg=(f"\n"
                     f"in  : {in_grid}\n"
                     f"out : {out_grid}\n"
                     f"pred: {pred_grid}\n"),
            )

    def get_train_program(self, task_num):
        if task_num == 0:
            return lambda x: F._kronecker(F._color_i_to_j(x)(0)(-1))(x)
        elif task_num == 86:
            return lambda x: F._rotate_cw(F._rotate_cw(x))
        elif task_num == 139:
            return lambda x: F._rotate_ccw(F._rotate_ccw(x))
        elif task_num == 222:
            return lambda x: F._inflate(x)(3)
        elif task_num == 379:
            return lambda x: F._rotate_ccw(x)

        return None

    def test_on_train_tasks(self):
        total_solved = 0

        for task_num in range(400):
            program = self.get_train_program(task_num)
            if program is not None:
                with self.subTest(task_num=task_num):
                    self.check_arc_train_task(task_num, program)
                    total_solved += 1

        print(f"Solved {total_solved} ARC train tasks.")


class PrimitiveTests(unittest.TestCase):
    def inflate_deflate_test(self):
        rng = np.random.default_rng()
        for scale in range(1, 5):
            original = rng.integers(0, 10, (3, 3))
            upscaled = F._inflate(original)(scale)
            downscaled = F.deflate(upscaled)(scale)

            self.assertEqual(
                original, 
                downscaled,
                msg=(f"\n"
                     f"upscaled  : {original}\n"
                     f"downscaled : {downscaled}\n"),
            )
