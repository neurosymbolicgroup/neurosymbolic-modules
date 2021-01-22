from rl.operations import (forward_op, constant_op, inverse_op,
                           cond_inverse_op, Op)
from typing import Callable, List, Tuple, Dict

import bidir.primitives.functions as F
import bidir.primitives.inverse_functions as F2
from bidir.primitives.types import COLORS


FUNCTIONS: List[Callable] = [
    F.hstack_pair,
    F.hflip,
    F.vflip,
    F.vstack_pair,
    F.rotate_cw,
    F.rotate_ccw,
    F.rows,
    F.columns,
    F.hstack,
    F.vstack,
    F.block,
]

FORWARD_FUNCTION_OPS = [forward_op(fn) for fn in FUNCTIONS]

COLOR_OPS = [constant_op(c, name=f"{COLORS.name_of(c)}")
    for c in COLORS.ALL_COLORS]

BOOL_OPS = [constant_op(b) for b in [True, False]]

MAX_INT = 3
INT_OPS = [constant_op(i) for i in range(MAX_INT)]

FORWARD_OPS = FORWARD_FUNCTION_OPS + COLOR_OPS + BOOL_OPS + INT_OPS

# TODO: Should we move these defs into bidir.primitives.functions?
_FUNCTION_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.rotate_ccw, F.rotate_cw),
    (F.rotate_cw, F.rotate_ccw),
    (F.vflip, F.vflip),
    (F.hflip, F.hflip),
    (F.rows, F.vstack),
    (F.columns, F.hstack),
    (F.block, F2.block_inv),
]

INV_OPS = [inverse_op(fn, inverse_fn)
    for (fn, inverse_fn) in _FUNCTION_INV_PAIRS]

_FUNCTION_COND_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.vstack_pair, F2.vstack_pair_cond_inv),
    (F.hstack_pair, F2.hstack_pair_cond_inv),
    (F.inflate, F2.inflate_cond_inv)
]

COND_INV_OPS = [cond_inverse_op(fn, inverse_fn)
    for (fn, inverse_fn) in _FUNCTION_COND_INV_PAIRS]

ALL_OPS = FORWARD_OPS + INV_OPS + COND_INV_OPS

assert len(set(op.fn.name for op in FORWARD_OPS)) == len(FORWARD_OPS), (
        "duplicate op name")
assert len(set(op.fn.name for op in INV_OPS)) == len(INV_OPS), (
        "duplicate inverse op name")
assert len(set(op.fn.name for op in COND_INV_OPS)) == len(COND_INV_OPS), (
        "duplicate cond. inverse op name")

FORWARD_DICT: Dict[str, Op] = {op.fn.name: op for op in FORWARD_OPS}
INV_DICT: Dict[str, Op] = {op.fn.name + '_inv': op for op in INV_OPS}
COND_INV_DICT: Dict[str, Op] = {
    op.fn.name + '_cond_inv': op
    for op in COND_INV_OPS
}

OP_DICT: Dict[str, Op] = {**FORWARD_DICT, **INV_DICT, **COND_INV_DICT}

if __name__ == '__main__':
    print(OP_DICT)
    # print(OP_DICT['3'].fn)
