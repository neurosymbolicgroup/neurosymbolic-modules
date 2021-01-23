from rl.operations import Op, ForwardOp, InverseOp, CondInverseOp, ConstantOp
from typing import Callable, List, Tuple, Dict

import bidir.primitives.functions as F
import bidir.primitives.inverse_functions as F2
from bidir.primitives.types import COLORS


def tuple_return(f: Callable):
    '''
    Inverse functions should take in the output and return a tuple of input
    arguments.
    This function is useful if the inverse function is already implemented as a
    forward function, and we simply want it to return a tuple with a single
    element instead of a single value.
    For example, the proper inverse of rotate_cw is tuple_return(rotate_ccw).
    '''
    return lambda x: (f(x),)


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

FORWARD_OPS = [ForwardOp(fn) for fn in FUNCTIONS]

COLOR_OPS = [ConstantOp(c, name=f"{COLORS.name_of(c)}")
             for c in COLORS.ALL_COLORS]
BOOL_OPS = [ConstantOp(b) for b in [True, False]]
MAX_INT = 3
INT_OPS = [ConstantOp(i) for i in range(MAX_INT)]
CONSTANT_OPS = COLOR_OPS + BOOL_OPS + INT_OPS

# TODO: Should we move these defs into bidir.primitives.functions?
_FUNCTION_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.rotate_ccw, tuple_return(F.rotate_cw)),
    (F.rotate_cw, tuple_return(F.rotate_ccw)),
    (F.vflip, tuple_return(F.vflip)),
    (F.hflip, tuple_return(F.hflip)),
    (F.rows, tuple_return(F.vstack)),
    (F.columns, tuple_return(F.hstack)),
    (F.block, F2.block_inv),
]

INV_OPS = [InverseOp(fn, inverse_fn)
           for (fn, inverse_fn) in _FUNCTION_INV_PAIRS]

_FUNCTION_COND_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.vstack_pair, F2.vstack_pair_cond_inv),
    (F.hstack_pair, F2.hstack_pair_cond_inv),
    (F.inflate, F2.inflate_cond_inv)
]

COND_INV_OPS = [CondInverseOp(fn, inverse_fn)
                for (fn, inverse_fn) in _FUNCTION_COND_INV_PAIRS]

ALL_OPS = FORWARD_OPS + CONSTANT_OPS + INV_OPS + COND_INV_OPS

assert len(set(op.name for op in ALL_OPS)) == len(ALL_OPS), (
        "duplicate op name")

OP_DICT: Dict[str, Op] = {op.name: op for op in ALL_OPS}

if __name__ == '__main__':
    print(OP_DICT)
    # print(OP_DICT['3'].fn)
