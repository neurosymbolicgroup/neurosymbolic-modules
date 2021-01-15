import dreamcoder.domains.arc.bidir.primitives.functions as F
from dreamcoder.domains.arc.bidir.primitives.types import (
    Color,
    Grid,
    Tuple,
    Callable,

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
import typing
from typing import Callable, List, Tuple



class ForwardOp:
    def __init__(op_name, op, arg_types, return_type):
        self.op_name = op_name
        self.op = op
        self.arg_types = types
        self.return_type = return_type

    def function_op(op):
        """
        Creates a ForwardOp for a given function. Infers types
        from type hints, so the op needs to be implemented with type hints.
        """
        types = typing.get_type_hints(op)
        if len(types) == 0:
            raise ValueError('Operation provided does not use type hints, '
                    + 'which we use when choosing actions.')

        return_type = types['return']
        # list of (arg_name, arg_type) tuples whose length equals the number of
        # arguments
        arg_types = list(types.items())[0:-1]
        name = op.__name__
        return ForwardOp(name=name, op=op, arg_types=arg_types,
                return_type=return_type)

    def constant_op(name, value, value_type):
        """
        Creates a ForwardOp for a constant value. This just
        makes a function which takes zero arguments, and has the output type of
        the value type given.
        """
        # these will just be functions with zero arguments.
        op = lambda: value
        arg_types = []
        return_type = value_type
        return ForwardOp(name=name, op=op, arg_types=arg_types,
                return_type=return_type)


class BackwardOp:
    """
    This has all the same attributes as a ForwardOp, with the addition of an
    inverse_op function which returns a tuple of input arguments for the
    function. For now, let's assume inverse ops are one-to-one.
    """
    def __init__(op_name, op, arg_types, return_type, inverse_op):
        self.op_name = op_name
        self.op = op
        self.arg_types = types
        self.return_type = return_type
        self.inverse_op = inverse_op

    def function_op(op, inverse_op):
        """
        Creates a BackwardOp for a given function. Infers types
        from type hints, so the op needs to be implemented with type hints.
        inverse_op returns a tuple of input arguments for the function, and
        doesn't need any type hints.
        """
        types = typing.get_type_hints(op)
        if len(types) == 0:
            raise ValueError('Operation provided does not use type hints, '
                    + 'which we use when choosing actions.')

        return_type = types['return']
        # list of (arg_name, arg_type) tuples whose length equals the number of
        # arguments
        arg_types = list(types.items())[0:-1]
        name = op.__name__

        return BackwardOp(name=name, op=op, arg_types=arg_types,
                return_type=return_type, inverse_op=inverse_op)


functions = [
    F.color_i_to_j,
    F.rotate_ccw,
    F.rotate_cw,
    F.inflate,
    F.deflate,
    F.kronecker,
    F.crop,
    F.set_bg,
    F.unset_bg,
    F.size,
    F.area,
    F.get_color,
    F.color_in,
    F.filter_color,
    F.top_half,
    F.vflip,
    F.hflip,
    F.empty_grid,
    F.vstack,
    F.hstack,
    F.vstack_pair,
    F.hstack_pair,
    F.rows,
    F.columns,
    F.overlay,
    F.overlay_pair,
]

forward_function_ops = [ForwardOp.function_op(fn) for fn in functions]

colors = [
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
]

color_ops = [ForwardOp.constant_op(color, Color) for color in colors]

# stick to small ints for now?
ints = [i for i in range(3)]

int_ops = [ForwardOp.constant_op(i, int) for i in ints]

bools = [True, False]

bool_ops = [ForwardOp.constant_op(b, bool) for b in bools]

all_forward_ops = forward_function_ops, color_ops, int_ops, bool_ops

# sticking to one-to-one functions for now.
function_inverse_pairs = [
    (F.rotate_ccw, F.rotate_cw),
    (F.rotate_cw, F.rotate_ccw),
    (F.vflip, F.vflip),
    (F.hflip, F.hflip),
    (F.rows, F.vstack),
    (F.columns, F.hstack),
]

all_backward_ops = [BackwardOp.function_op(op, inverse)
        for op, inverse in function_inverse_pairs]


