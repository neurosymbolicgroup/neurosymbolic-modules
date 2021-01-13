import unittest
import numpy as np

from dreamcoder.domains.arc.bidir.task_utils import get_task_grid_pairs
from dreamcoder.domains.arc.bidir.types import Grid, COLORS
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
        elif task_num == 30:
            return lambda x: F._crop(F._set_bg(x)(-1))
        elif task_num == 38:
            def solve(x):
                obj = F._crop(F._set_bg(x)(-1))
                top_half = F._top_half(obj)
                top_left = F._rotate_cw(F._top_half(F._rotate_ccw(top_half)))
                return top_left

            return solve
        elif task_num == 56:
            def solve(x):
                obj = F._crop(F._set_bg(x)(-1))
                empty_grid = F._empty_grid(1)(2)
                hblock = F._color_i_to_j(empty_grid)(Grid.BACKGROUND_COLOR)(1)
                out = F._kronecker(obj, hblock)
                return out

            return solve
        elif task_num == 86:
            return lambda x: F._rotate_cw(F._rotate_cw(x))
        elif task_num == 128:
            return lambda x: F._color_in(x)(F._color(x))
        elif task_num == 139:
            return lambda x: F._rotate_ccw(F._rotate_ccw(x))
        elif task_num == 152:
            return lambda x: F._vflip(x)
        elif task_num == 194:
            def solve(x):
                deflated = F._deflate(F._crop(F._set_bg(x)(-1)))
                return lambda x: F._kronecker(deflated)(deflated)
            return solve
        elif task_num == 222:
            return lambda x: F._inflate(x)(3)
        # waiting until overlay is implemented
        # elif task_num == 228:
        #     def solve(x):
        #         color = F._get_color(x)
        #         just_color = F._filter_color(x)(color)
        #         greyed_out = F._color_in(x)(COLORS['grey'])
        #         out = F._overlay(just_color)(greyed_out)
        #         return out

        #     return solve
        elif task_num == 268:
            return lambda x: F.inflate(x)(F._area(F._set_bg(x)(-1)))
        elif task_num == 275:
            return lambda x: F._color_i_to_j(x)(COLORS['pink'], COLORS['red'])
        elif task_num == 289:
            def solve(x):
                obj = F._crop(F._set_bg(x)(COLORS['black']))
                color = F._get_color(obj)
                obj = F._set_bg(x)(color)
                color2 = F._get_color(obj)
                obj = F._color_i_to_j(obj)(color2)(color)
                obj = F._color_i_to_j(obj)(Grid.BACKGROUND_COLOR)(color2)
                return obj

            return solve
        elif task_num == 303:
            def solve(x):
                filtered = F._filter_color(x)(F._get_color(x))
                return lambda x: F._kronecker(x)(filtered)

            return solve
        elif task_num == 306:
            return lambda x: F._inflate(x)(2)
        elif task_num == 308:
            orange = COLORS['orange']
            grey = COLORS['grey']
            return lambda x: F._color_i_to_j(x)(orange)(grey)
        elif task_num == 379:
            return lambda x: F._rotate_ccw(x)
        elif task_num == 383:
            return lambda x: F._inflate(F._crop(F._set_bg(x)(-1)))
        elif task_num == 388:
            def solve(x):
                obj = F._set_bg(x)(COLORS['grey'])
                color = F._get_color(obj)
                obj = F._color_i_to_j(obj)(color)(COLORS['black'])
                obj = F._color_i_to_j(obj)(Grid.BACKGROUND_COLOR)(color)
                return obj

            return solve
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
            downscaled = F._deflate(upscaled)(scale)

            self.assertEqual(original, downscaled)

    def size_test(self):
        grid = Grid(np.ones(3, 7))
        self.assertEqual(F._size(grid), 3 * 7)
