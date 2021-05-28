from typing import Callable, List, Tuple

import bidir.primitives.functions as F
import bidir.primitives.inverse_functions as F2
from bidir.primitives.types import Color
from rl.ops.operations import ForwardOp, InverseOp, CondInverseOp, ConstantOp
from rl.ops.utils import tuple_return

FUNCTIONS: List[Callable] = [
    F.get_color,
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
    F.set_bg,
    F.unset_bg,
    F.crop,
    F.kronecker,
    F.top_half,
    F.colors,
    F.length,
    F.inflate,
    F.inflate2,
    F.inflate3,
    F.deflate2,
    F.deflate3,
    F.color_i_to_j,
]

FORWARD_OPS = [ForwardOp(fn) for fn in FUNCTIONS]

COLOR_OPS = [ConstantOp(c) for c in Color]
BOOL_OPS = [ConstantOp(b) for b in [True, False]]
MAX_INT = 3
INT_OPS = [ConstantOp(i) for i in range(MAX_INT + 1)]
CONSTANT_OPS = COLOR_OPS + BOOL_OPS + INT_OPS

# TODO: Should we move these defs into bidir.primitives.functions?
"""
Note that every invertible function has two InverseOps associated with it
So if you want to apply rotate_cw, you'll call the first InverseOp we have below
If you want to apply rotate_ccw, you'll call the second InverseOp we have below
"""
_FUNCTION_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.rotate_ccw, tuple_return(F.rotate_cw)),
    (F.rotate_cw, tuple_return(F.rotate_ccw)),
    (F.vflip, tuple_return(F.vflip)),
    (F.hflip, tuple_return(F.hflip)),
    (F.rows, tuple_return(F.vstack)),
    (F.columns, tuple_return(F.hstack)),
    # (F.block, F2.block_inv),
    (F.inflate2, tuple_return(F.deflate2)),
    (F.inflate3, tuple_return(F.deflate3)),
]

INV_OPS = [
    InverseOp(fn, inverse_fn) for (fn, inverse_fn) in _FUNCTION_INV_PAIRS
]

_FUNCTION_COND_INV_PAIRS: List[Tuple[Callable, Callable, List[bool]]] = [
    (F.vstack_pair, F2.vstack_pair_cond_inv_top, [True, False]),
    (F.vstack_pair, F2.vstack_pair_cond_inv_bottom, [False, True]),
    (F.hstack_pair, F2.hstack_pair_cond_inv_left, [True, False]),
    (F.hstack_pair, F2.hstack_pair_cond_inv_right, [False, True]),
    # (F.inflate, F2.inflate_cond_inv),
]

COND_INV_OPS = [
    CondInverseOp(fn, inverse_fn, expects_cond)
    for (fn, inverse_fn, expects_cond) in _FUNCTION_COND_INV_PAIRS
]

ALL_OPS = FORWARD_OPS + CONSTANT_OPS + INV_OPS + COND_INV_OPS  # type: ignore

_duplicates = [op for op in ALL_OPS if sum(op == i for i in ALL_OPS) > 1]
if not len(set(op.name for op in ALL_OPS)) == len(ALL_OPS):
    _duplicates = [
        op for op in ALL_OPS if sum(op.name == i.name for i in ALL_OPS) > 1
    ]
    assert False, f"duplicate op names: {_duplicates}"

OP_DICT = {op.name: op for op in ALL_OPS}

BIDIR_GRID_OP_NAMES = [
    'hflip',
    'hflip_inv',
    'vflip',
    'vflip_inv',
    'hstack_pair',
    'hstack_pair_cond_inv_left',
    'hstack_pair_cond_inv_right',
    'vstack_pair',
    'vstack_pair_cond_inv_top',
    'vstack_pair_cond_inv_bottom',
    'rotate_cw',
    'rotate_cw_inv',
    'rotate_ccw',
    'rotate_ccw_inv',
    'inflate2',
    'inflate2_inv',
    'inflate3',
    'inflate3_inv',
    'top_half',
]

BIDIR_GRID_OPS = [OP_DICT[n] for n in BIDIR_GRID_OP_NAMES]
FW_GRID_OPS = [op for op in BIDIR_GRID_OPS if isinstance(op, ForwardOp)]

# Takes grids as inputs and outputs a grid
# Note: if you rearrange these, it might mess up old NN training runs, which
# rely on having the same order of ops
# deprecated: want to start using FW_GRID_OPS instead
GRID_FUNCTIONS: List[Callable] = [
    F.hstack_pair,
    F.hflip,
    F.vflip,
    F.vstack_pair,
    F.rotate_cw,
    F.rotate_ccw,
    F.top_half,
]

GRID_OPS = [ForwardOp(fn) for fn in GRID_FUNCTIONS]

# only those with type Grid -> Grid aka arity 1
# F.hflip, F.vflip, F.rotate_cw, F.rorate_ccw, F.top_half
GRID_OPS_ARITY_ONE = [op for op in GRID_OPS if op.arity == 1]
