import unittest
import numpy as np
from functools import reduce

from bidir.task_utils import get_task_grid_pairs
from bidir.primitives.types import (
    Grid,
    BLACK,
    BLUE,
    RED,
    GREEN,
    YELLOW,
    GREY,
    PINK,
    ORANGE,
    CYAN,
    MAROON,
    BACKGROUND_COLOR,
)
import bidir.primitives.functions as F


class TestOnTasks(unittest.TestCase):
    def check_arc_train_task(self, task_num, program):
        grid_pairs = get_task_grid_pairs(task_num, train=True)
        for in_grid, out_grid in grid_pairs:
            pred_grid = program(in_grid)

            self.assertEqual(
                pred_grid,
                out_grid,
                msg=(f"\n"
                     f"task number: {task_num}\n"
                     f"in  : {in_grid}\n"
                     f"out : {out_grid}\n"
                     f"pred: {pred_grid}\n"),
            )

    def get_train_program(self, task_num):
        # yapf: disable
        if task_num == 0:
            def solve(x):
                obj = F.set_bg(x, BLACK)
                obj = F.kronecker(obj, obj)
                return F.unset_bg(obj, BLACK)
            return solve
        elif task_num == 30:
            return lambda x: F.unset_bg(F.crop(F.set_bg(x, BLACK)), BLACK)
        elif task_num == 31:
            def solve(x):
                columns = F.columns(x)

                def vblock(height, color):
                    return F.unset_bg(F.empty_grid(height, 1), color)

                def f1(col):
                    col = F.set_bg(col, BLACK)
                    return vblock(F.area(col), F.get_color(col))

                def f2(col):
                    col = F.filter_color(col, BLACK)
                    return vblock(F.area(col), BLACK)

                f3 = lambda col: F.vstack_pair(f2(col), f1(col))
                blocks = F.map_fn(f3, columns)
                return F.hstack(blocks)
            return solve
        elif task_num == 38:
            def solve(x):
                obj = F.crop(F.set_bg(x, BLACK))
                top_half = F.top_half(obj)
                top_left = F.rotate_ccw(F.top_half(F.rotate_cw(top_half)))
                return F.unset_bg(top_left, BLACK)
            return solve
        elif task_num == 56:
            def solve(x):
                obj = F.crop(F.set_bg(x, BLACK))
                empty_grid = F.empty_grid(1, 2)
                hblock = F.unset_bg(empty_grid, BLACK)
                out = F.kronecker(hblock, obj)
                return F.unset_bg(out, BLACK)
            return solve
        elif task_num == 82:
            def solve(x):
                top = F.hstack_pair(x, F.hflip(x))
                bottom = F.hstack_pair(F.vflip(x), F.hflip(F.vflip(x)))
                return F.vstack_pair(top, bottom)
            return solve
        elif task_num == 86:
            return lambda x: F.rotate_cw(F.rotate_cw(x))
        elif task_num == 99:
            def solve(x):
                block = F.empty_grid(2, 2)
                return F.unset_bg(block, F.get_color(F.set_bg(x, BLACK)))
            return solve
        elif task_num == 105:
            def solve(x):
                x1 = F.rotate_cw(x)
                x2 = F.rotate_cw(x1)
                x3 = F.rotate_cw(x2)
                top = F.hstack_pair(x, x1)
                bottom = F.hstack_pair(x3, x2)
                return F.vstack_pair(top, bottom)
            return solve
        elif task_num == 112:
            def solve(x):
                x = F.set_bg(x, BLACK)
                x = F.overlay_pair(x, F.vflip(x))
                return F.unset_bg(x, BLACK)
            return solve
        elif task_num == 115:
            return lambda x: F.vstack_pair(F.vflip(x), x)
        elif task_num == 128:
            return lambda x: F.color_in(x, F.get_color(x))
        elif task_num == 139:
            return lambda x: F.rotate_ccw(F.rotate_ccw(x))
        elif task_num == 141:
            def solve(x):
                top = F.hstack_pair(x, F.hflip(x))
                bottom = F.hstack_pair(F.vflip(x), F.hflip(F.vflip(x)))
                return F.vstack_pair(top, bottom)
            return solve
        elif task_num == 149:
            return lambda x: F.hflip(x)
        elif task_num == 151:
            def solve(x):
                top = F.hstack_pair(x, F.hflip(x))
                bottom = F.hstack_pair(F.vflip(x), F.hflip(F.vflip(x)))
                return F.vstack_pair(top, bottom)
            return solve
        elif task_num == 154:
            # solving more circuitously to test map function
            def solve(x):
                cols = F.columns(x)
                fn = lambda col: F.vflip(col)
                flipped_cols = F.map_fn(fn, cols)
                out = F.hstack(flipped_cols)
                return out
            return solve
        elif task_num == 163:
            return lambda x: F.hstack_pair(x, F.hflip(x))
        elif task_num == 171:
            return lambda x: F.vstack_pair(x, F.vflip(x))
        elif task_num == 176:
            return lambda x: F.hflip(F.crop(F.set_bg(x, BLACK)))
        elif task_num == 178:
            return lambda x: F.hflip(F.rotate_cw(x))
        elif task_num == 194:
            def solve(x):
                deflated = F.deflate(F.crop(F.set_bg(x, BLACK)), 3)
                return F.unset_bg(F.kronecker(deflated, deflated), BLACK)
            return solve
        elif task_num == 209:
            return lambda x: F.vstack_pair(x, F.vflip(x))
        elif task_num == 210:
            def solve(x):
                y = F.hstack_pair(F.hflip(x), x)
                z = F.vflip(y)
                return F.vstack_pair(F.vstack_pair(z, y), z)
            return solve
        elif task_num == 216:
            def solve(x):
                obj = F.set_bg(x, BLACK)
                obj = F.crop(obj)
                obj = F.kronecker(obj, obj)
                return F.unset_bg(obj, BLACK)
            return solve
        elif task_num == 222:
            return lambda x: F.inflate(x, 3)
        elif task_num == 228:
            def solve(x):
                color = F.get_color(x)
                just_color = F.filter_color(x, color)
                greyed_out = F.color_in(x, GREY)
                out = F.overlay_pair(just_color, greyed_out)
                return out
            return solve
        elif task_num == 240:
            return lambda x: F.hflip(F.rotate_cw(x))
        elif task_num == 248:
            return lambda x: F.hstack_pair(x, x)
        elif task_num == 256:
            def solve(x):
                x = F.set_bg(x, BLUE)
                top = F.top_half(x)
                bottom = F.vflip(F.top_half(F.vflip(x)))

                def left_half(x):
                    return F.rotate_ccw(F.top_half(F.rotate_cw(x)))

                def right_half(x):
                    return F.rotate_cw(F.top_half(F.rotate_ccw(x)))

                top_left = left_half(top)
                top_right = right_half(top)
                bottom_left = left_half(bottom)
                bottom_right = right_half(bottom)

                crop_tl = F.crop(top_left)
                crop_tr = F.crop(top_right)
                crop_bl = F.crop(bottom_left)
                crop_br = F.crop(bottom_right)

                crop_tl = F.set_bg(crop_tl, BLACK)
                crop_tr = F.set_bg(crop_tr, BLACK)
                crop_bl = F.set_bg(crop_bl, BLACK)
                crop_br = F.set_bg(crop_br, BLACK)

                out = F.overlay_pair(crop_tl, crop_tr)
                out = F.overlay_pair(out, crop_bl)
                out = F.overlay_pair(out, crop_br)

                out = F.unset_bg(out, BLACK)
                return out

            return solve
        elif task_num == 258:
            return lambda x: F.unset_bg(F.crop(F.set_bg(x, BLUE)), BLACK)
        elif task_num == 268:
            def solve(x):
                out = F.inflate(x, F.area(F.set_bg(x, BLACK)))
                return F.unset_bg(out, BLACK)
            return solve
        elif task_num == 275:
            return lambda x: F.color_i_to_j(x, PINK, RED)
        elif task_num == 289:
            def solve(x):
                obj = F.crop(F.set_bg(x, BLACK))
                color = F.get_color(obj)
                obj = F.set_bg(obj, color)
                color2 = F.get_color(obj)
                obj = F.color_i_to_j(obj, color2, color)
                obj = F.color_i_to_j(obj, BACKGROUND_COLOR, color2)
                return obj
            return solve
        elif task_num == 299:
            def solve(x):
                x = F.set_bg(x, BLACK)
                color = F.get_color(x)
                x = F.filter_color(x, color)
                return F.unset_bg(F.crop(x), BLACK)
            return solve
        elif task_num == 303:
            def solve(x):
                filtered = F.filter_color(x, F.get_color(x))
                return F.unset_bg(F.kronecker(filtered, x), BLACK)
            return solve
        elif task_num == 306:
            return lambda x: F.inflate(x, 2)
        elif task_num == 308:
            return lambda x: F.color_i_to_j(x, ORANGE, GREY)
        elif task_num == 310:
            return lambda x: F.hstack_pair(x, F.hflip(x))
        elif task_num == 336:
            def solve(x):
                x = F.color_i_to_j(x, CYAN, BLACK)
                x = F.color_i_to_j(x, GREY, CYAN)
                x = F.color_i_to_j(x, BLACK, GREY)
                return x
            return solve
        elif task_num == 338:
            def solve(x):
                x = F.set_bg(x, BLACK)
                a = F.area(x)
                c = F.get_color(x)
                return F.unset_bg(F.empty_grid(1, a), c)
            return solve
        elif task_num == 359:
            def solve(x):
                left = F.rotate_ccw(F.top_half(F.rotate_cw(x)))
                left = F.crop(F.set_bg(left, GREY))
                right = F.rotate_cw(F.top_half(F.rotate_ccw(x)))
                right = F.crop(F.set_bg(right, GREY))
                left = F.set_bg(left, BLACK)
                right = F.set_bg(right, BLACK)
                out = F.overlay_pair(left, F.hflip(right))
                return F.unset_bg(out, BLACK)
            return solve
        elif task_num == 379:
            return lambda x: F.rotate_ccw(x)
        elif task_num == 383:
            def solve(x):
                obj = F.inflate(F.crop(F.set_bg(x, BLACK)), 2)
                return F.unset_bg(obj, BLACK)
            return solve
        elif task_num == 384:
            def solve(x):
                x = F.set_bg(x, BLACK)
                x = F.overlay_pair(x, F.vflip(x))
                return F.unset_bg(x, BLACK)
            return solve
        elif task_num == 388:
            def solve(x):
                obj = F.set_bg(x, GREY)
                color = F.get_color(obj)
                obj = F.color_i_to_j(obj, color, BLACK)
                obj = F.color_i_to_j(obj, BACKGROUND_COLOR, color)
                return obj
            return solve
        else:
            return "No program"
        # yapf: enable

    def test_on_train_tasks(self):
        total_solved = 0

        for task_num in range(400):
            program = self.get_train_program(task_num)
            if program == "No program":
                continue

            with self.subTest(task_num=task_num):
                self.assertNotEqual(
                    program, None,
                    f"program for {task_num} is None. Did you forget to 'return solve'?"
                )
                self.check_arc_train_task(task_num, program)
                total_solved += 1

        print(f"\nSolved {total_solved} ARC train tasks")


class PrimitiveTests(unittest.TestCase):
    def check_grids_equal(self, target, pred):
        self.assertEqual(
            target,
            pred,
            msg=(f"\n"
                 f"target: {target}\n"
                 f"pred  : {pred}\n"),
        )

    def test_inflate_deflate(self):
        rng = np.random.default_rng()
        for scale in range(1, 5):
            original = Grid(rng.integers(0, 10, (3, 3)))
            upscaled = F.inflate(original, scale)
            downscaled = F.deflate(upscaled, scale)

            self.check_grids_equal(original, downscaled)

    def test_size(self):
        grid = Grid(np.ones((3, 7), dtype=int))
        self.assertEqual(F.size(grid), 3 * 7)

    def test_stack_padding(self):
        grid_pairs = get_task_grid_pairs(task_num=347, train=True)
        grid = grid_pairs[0][1]  # first example's output

        def block(height, color):
            return Grid(np.full((height, 1), color))

        blocks = [
            block(1, CYAN),
            block(2, ORANGE),
            block(3, CYAN),
            block(4, ORANGE),
            block(3, CYAN),
            block(2, ORANGE),
            block(1, CYAN)
        ]

        # tests horizontal padding both ways
        stacked = reduce(lambda b1, b2: F.hstack_pair(b1, b2), blocks)
        # tests vertical padding with bottom smaller
        stacked = F.vstack_pair(stacked, Grid(np.full((1, 1), BLACK)))
        out = F.unset_bg(stacked, BLACK)
        self.check_grids_equal(out, grid)

    def test_stack_padding2(self):
        grid_pairs = get_task_grid_pairs(task_num=347, train=True)
        grid = grid_pairs[0][1]  # first example's output

        def block(height, color):
            return Grid(np.full((height, 1), color))

        # tests vertical padding with bottom larger
        grid = grid_pairs[0][0]
        bottom = Grid(np.full((1, 7), BLACK))
        top = block(4, ORANGE)
        top = F.hstack_pair(Grid(np.full((4, 3), BLACK)), top)
        stacked = F.vstack_pair(top, bottom)
        out = F.unset_bg(stacked, BLACK)
        self.check_grids_equal(out, grid)

    def test_rows_and_columns(self):
        for task_num in range(10):
            grid_pairs = get_task_grid_pairs(task_num, train=True)
            for i, o in grid_pairs:
                self.check_grids_equal(i, F.vstack(F.rows(i)))
                self.check_grids_equal(i, F.hstack(F.columns(i)))

    def test_rows_and_columns2(self):
        def solve(x):
            rows = F.rows(F.crop(F.set_bg(x, BLACK)))
            rows = rows[0:3]
            grid = F.vstack(rows)
            cols = F.columns(grid)
            cols = cols[0:3]
            return F.unset_bg(F.hstack(cols), BLACK)

        grid_pairs = get_task_grid_pairs(38, train=True)
        for in_grid, out_grid in grid_pairs:
            pred_grid = solve(in_grid)

            self.assertEqual(
                pred_grid,
                out_grid,
                msg=(f"\n"
                     f"task number: {38}\n"
                     f"in  : {in_grid}\n"
                     f"out : {out_grid}\n"
                     f"pred: {pred_grid}\n"),
            )
