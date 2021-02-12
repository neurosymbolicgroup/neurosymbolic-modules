import unittest
import random
from typing import Tuple, Callable, Any
from bidir.utils import SynthError

import bidir.task_utils as utils
from bidir.primitives.types import Grid, COLORS
import bidir.primitives.functions as F
import bidir.primitives.inverse_functions as F2

NUM_RANDOM_TESTS = 1


def get_grids():
    grids = []
    for i in range(400):
        task_id = utils.num_to_id(i)

        task_dict = utils.load_task(task_id,
                                    task_path="data/ARC/data/training/")
        for x in task_dict["train"] + task_dict["test"]:
            grids.append(Grid(x["input"]))
            grids.append(Grid(x["output"]))

    return grids


class Sampler:
    def __init__(self):
        # self.grids = get_grids()
        self.grids = [g for g in get_grids() if max(g.arr.shape) < 4]

    def sample(self, choices):
        return random.sample(choices, k=1)[0]

    def sample_grid(self, filter_fn=lambda x: True):
        grids = [g for g in self.grids if filter_fn(g)]
        return self.sample(grids)

    def sample_color(self):
        return self.sample(COLORS.ALL_COLORS)

    def sample_int(self, min_val, max_val):
        return self.sample(list(range(min_val, max_val)))

    def sample_bool(self):
        return self.sample([True, False])


class InverseFunctionTests(unittest.TestCase):
    def setUp(self):
        # used to sample random grids, etc.
        self.sampler = Sampler()
        # number of random tests to do per function

    def repeated_test(self,
                      test_fn,
                      n_tests=NUM_RANDOM_TESTS,
                      max_failures_per_test=2):
        failures = 0
        for test_i in range(n_tests):
            with self.subTest(test_i=test_i):
                worked = False
                while not worked:
                    worked = True
                    try:
                        test_fn()
                    # some error during creation, e.g. grid too large
                    # doesn't mean the inverse doesn't work, just means the
                    # randomly sampled grids aren't correct
                    except SynthError:
                        worked = False
                        failures += 1
                        if failures > max_failures_per_test * n_tests:
                            self.assertTrue(False, 'too many failures')

    def check_inverse(
        self,
        forward_fn: Callable,
        inverse_fn: Callable[[Any], Tuple[Any, ...]],
        args: Tuple[Any, ...],
    ):
        out = forward_fn(*args)
        in_args = inverse_fn(out)
        self.assertEqual(
            args,
            in_args,
            msg=(f"Inputs are:\t{args}\n"
                 f"Inverse gave:\t{in_args}"),
        )

    def check_cond_inverse(
        self,
        forward_fn: Callable,
        inverse_fn: Callable,
        args: Tuple[Any, ...],
        inverse_args: Tuple[Any, ...],
        expects_cond: Tuple[bool, ...],
    ):
        out = forward_fn(*args)
        # why oh why did I decide to use tuples for everything lol
        inverse_args2 = tuple([out] + list(inverse_args))
        new_inputs = inverse_fn(*inverse_args2)

        cond_count = 0
        new_count = 0
        full_args = []
        arg_ix = 0
        # compile input values from two locations: the new computed values, and
        # the values provided as conditional inputs
        while arg_ix < len(args):
            if expects_cond[arg_ix]:
                # value already exists.
                full_args.append(inverse_args[cond_count])
                cond_count += 1
            else:
                # value was computed via cond inverse
                full_args.append(new_inputs[new_count])
                new_count += 1

            arg_ix += 1
            self.assertEqual(new_count, arg_ix - cond_count)

        self.assertEqual(cond_count, sum(expects_cond))
        self.assertEqual(new_count, len(new_inputs))

        self.assertEqual(
            args,
            tuple(full_args),
            msg=(f"Inputs are:\t{args}\n"
                 f"Inverse gave:\t{full_args}"),
        )

    def vstack_pair_top_test(self):
        top = self.sampler.sample_grid()
        # make sure has the same width as top
        bottom = self.sampler.sample_grid(
            lambda g: g.arr.shape[1] == top.arr.shape[1])

        self.check_cond_inverse(
            F.vstack_pair,
            F2.vstack_pair_cond_inv_top,
            (top, bottom),
            (top, ),
            (True, False),
        )

    def vstack_pair_bottom_test(self):
        bottom = self.sampler.sample_grid()
        # make sure has the same width as top
        top = self.sampler.sample_grid(
            lambda g: g.arr.shape[1] == bottom.arr.shape[1])

        self.check_cond_inverse(
            F.vstack_pair,
            F2.vstack_pair_cond_inv_bottom,
            (top, bottom),
            (bottom, ),
            (False, True),
        )

    def test_vstack_pair(self):
        self.repeated_test(self.vstack_pair_top_test)
        self.repeated_test(self.vstack_pair_bottom_test)

    def hstack_pair_left_test(self):
        left = self.sampler.sample_grid()
        # make sure the right has the same width as the left
        right = self.sampler.sample_grid(
            lambda g: g.arr.shape[0] == left.arr.shape[0])

        self.check_cond_inverse(
            F.hstack_pair,
            F2.hstack_pair_cond_inv_left,
            (left, right),
            (left, ),
            (True, False),
        )

    def hstack_pair_right_test(self):
        right = self.sampler.sample_grid()
        # make sure the right has the same width as the left
        left = self.sampler.sample_grid(
            lambda g: g.arr.shape[0] == right.arr.shape[0])

        self.check_cond_inverse(
            F.hstack_pair,
            F2.hstack_pair_cond_inv_right,
            (left, right),
            (right, ),
            (False, True),
        )

    def test_hstack_pair(self):
        self.repeated_test(self.hstack_pair_left_test)
        self.repeated_test(self.hstack_pair_right_test)

    def test_block(self):
        for i in range(NUM_RANDOM_TESTS):
            W = self.sampler.sample_int(1, 30)
            H = self.sampler.sample_int(1, 30)
            color = self.sampler.sample_color()
            self.check_inverse(F.block, F2.block_inv, (H, W, color))
