import unittest
import numpy as np
from functools import reduce

from bidir.task_utils import get_task_grid_pairs
from bidir.primitives.types import Grid, COLORS
import bidir.primitives.functions as F


class PrimitiveFunctionTests(unittest.TestCase):
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
            block(1, COLORS.CYAN),
            block(2, COLORS.ORANGE),
            block(3, COLORS.CYAN),
            block(4, COLORS.ORANGE),
            block(3, COLORS.CYAN),
            block(2, COLORS.ORANGE),
            block(1, COLORS.CYAN)
        ]

        # tests horizontal padding both ways
        stacked = reduce(lambda b1, b2: F.hstack_pair(b1, b2), blocks)
        # tests vertical padding with bottom smaller
        stacked = F.vstack_pair(stacked, Grid(np.full((1, 1), COLORS.BLACK)))
        out = F.unset_bg(stacked, COLORS.BLACK)
        self.check_grids_equal(out, grid)

    def test_stack_padding2(self):
        grid_pairs = get_task_grid_pairs(task_num=347, train=True)
        grid = grid_pairs[0][1]  # first example's output

        def block(height, color):
            return Grid(np.full((height, 1), color))

        # tests vertical padding with bottom larger
        grid = grid_pairs[0][0]
        bottom = Grid(np.full((1, 7), COLORS.BLACK))
        top = block(4, COLORS.ORANGE)
        top = F.hstack_pair(Grid(np.full((4, 3), COLORS.BLACK)), top)
        stacked = F.vstack_pair(top, bottom)
        out = F.unset_bg(stacked, COLORS.BLACK)
        self.check_grids_equal(out, grid)

    def test_rows_and_columns(self):
        for task_num in range(10):
            grid_pairs = get_task_grid_pairs(task_num, train=True)
            for i, o in grid_pairs:
                self.check_grids_equal(i, F.vstack(F.rows(i)))
                self.check_grids_equal(i, F.hstack(F.columns(i)))

    def test_rows_and_columns2(self):
        def solve(x):
            rows = F.rows(F.crop(F.set_bg(x, COLORS.BLACK)))
            rows = rows[0:3]
            grid = F.vstack(rows)
            cols = F.columns(grid)
            cols = cols[0:3]
            return F.unset_bg(F.hstack(cols), COLORS.BLACK)

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

    def test_sort_by_key(self):
        self.assertTupleEqual(
            F.sort_by_key(("a", "b", "c"), (2, 3, 1)),
            ("c", "a", "b"),
        )
