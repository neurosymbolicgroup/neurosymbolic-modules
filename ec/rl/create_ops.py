from rl.operations import forward_op, constant_op, inverse_op, cond_inverse_op
from typing import Callable, List, Tuple
import bidir.primitives.functions as F
import bidir.primitives.inverse_functions as F2
from bidir.primitives.types import COLORS


FUNCTIONS: List[Callable] = [
    F.hstack_pair,
    F.hflip,
    F.vflip,
    F.vstack_pair,
]

FORWARD_FUNCTION_OPS = [forward_op(fn) for fn in FUNCTIONS]

COLOR_OPS = [constant_op(c) for c in COLORS.ALL_COLORS]

BOOL_OPS = [constant_op(b) for b in [True, False]]

MAX_INT = 3
INT_OPS = [constant_op(i) for i in range(MAX_INT)]

FORWARD_OPS = FORWARD_FUNCTION_OPS + COLOR_OPS + BOOL_OPS + INT_OPS

# TODO: Should we move these defs into bidir.primitives.functions?
_FUNCTION_INVERSE_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.rotate_ccw, F.rotate_cw),
    (F.rotate_cw, F.rotate_ccw),
    (F.vflip, F.vflip),
    (F.hflip, F.hflip),
    (F.rows, F.vstack),
    (F.columns, F.hstack),
    (F.block, F2.block_inv),
]

INVERSE_OPS = [inverse_op(fn, inverse_fn)
    for (fn, inverse_fn) in _FUNCTION_INVERSE_PAIRS]

_FUNCTION_COND_INVERSE_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.vstack_pair, F2.vstack_pair_inv),
    (F.hstack_pair, F2.hstack_pair_inv),
]

COND_INVERSE_OPS = [cond_inverse_op(fn, inverse_fn)
    for (fn, inverse_fn) in _FUNCTION_COND_INVERSE_PAIRS]

ALL_OPS = FORWARD_OPS + INVERSE_OPS + COND_INVERSE_OPS

# an op will be a choice of OP along with at most MAX_ARITY arguments
MAX_ARITY = max(op.fn.arity for op in FORWARD_OPS)
N_OPS = len(ALL_OPS)
