import unittest
import numpy as np
from functools import reduce

from bidir.task_utils import get_arc_task_examples
from bidir.primitives.types import Grid, Color
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
        train_exs, test_exs = get_arc_task_examples(347, train=True)
        grid_pairs = train_exs + test_exs
        grid = grid_pairs[0][1]  # first example's output

        blocks = [
            F.block(1, 1, Color.CYAN),
            F.block(2, 1, Color.ORANGE),
            F.block(3, 1, Color.CYAN),
            F.block(4, 1, Color.ORANGE),
            F.block(3, 1, Color.CYAN),
            F.block(2, 1, Color.ORANGE),
            F.block(1, 1, Color.CYAN)
        ]

        # tests horizontal padding both ways
        stacked = reduce(lambda b1, b2: F.hstack_pair(b1, b2), blocks)
        # tests vertical padding with bottom smaller
        stacked = F.vstack_pair(stacked, F.block(1, 1, Color.BLACK))
        out = F.unset_bg(stacked, Color.BLACK)
        self.check_grids_equal(out, grid)

    def test_stack_padding2(self):
        train_exs, test_exs = get_arc_task_examples(347, train=True)
        grid_pairs = train_exs + test_exs
        grid = grid_pairs[0][1]  # first example's output

        # tests vertical padding with bottom larger
        grid = grid_pairs[0][0]
        bottom = F.block(1, 7, Color.BLACK)
        top = F.block(4, 1, Color.ORANGE)
        top = F.hstack_pair(F.block(4, 3, Color.BLACK), top)
        stacked = F.vstack_pair(top, bottom)
        out = F.unset_bg(stacked, Color.BLACK)
        self.check_grids_equal(out, grid)

    def test_rows_and_columns(self):
        for task_num in range(10):
            train_exs, test_exs = get_arc_task_examples(task_num, train=True)
            grid_pairs = train_exs + test_exs
            for i, o in grid_pairs:
                self.check_grids_equal(i, F.vstack(F.rows(i)))
                self.check_grids_equal(i, F.hstack(F.columns(i)))

    def test_rows_and_columns2(self):
        def solve(x):
            rows = F.rows(F.crop(F.set_bg(x, Color.BLACK)))
            rows = rows[0:3]
            grid = F.vstack(rows)
            cols = F.columns(grid)
            cols = cols[0:3]
            return F.unset_bg(F.hstack(cols), Color.BLACK)

        train_exs, test_exs = get_arc_task_examples(38, train=True)
        grid_pairs = train_exs + test_exs
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

    def test_filter_by_fn(self):
        def fn(c):
            return c <= 'b'

        self.assertTupleEqual(
            F.filter_by_fn(fn=fn, xs=("a", "b", "c", "b", "d")),
            ("a", "b", "b"),
        )

        def fn2(c):
            return c < 'a'

        self.assertTupleEqual(
            F.filter_by_fn(fn=fn2, xs=("a", "b", "c", "b", "d")),
            (),
        )

        def fn3(c):
            return c <= 'd'

        self.assertTupleEqual(
            F.filter_by_fn(fn=fn3, xs=("a", "b", "b", "d")),
            ("a", "b", "b", "d"),
        )

    def test_sort_by_key(self):
        self.assertTupleEqual(
            F.sort_by_key(("a", "b", "c"), (2, 3, 1)),
            ("c", "a", "b"),
        )
