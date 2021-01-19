import typing
from typing import Any, Callable, List, Tuple

import bidir.primitives.functions as F
from bidir.primitives.types import COLORS


class ForwardOp:
    def __init__(
        self,
        name: str,
        fn: Callable,
        arg_types: List,
        return_type,
    ):
        self.name = name
        self.fn = fn
        self.arg_types = arg_types
        self.num_args: int = len(self.arg_types)
        self.return_type = return_type

    @classmethod
    def op_from_fn(cls, fn: Callable):
        """
        Creates a ForwardOp for a given function. Infers types
        from type hints, so the op needs to be implemented with type hints.
        """
        types = typing.get_type_hints(fn)
        if len(types) == 0:
            raise ValueError(("Operation provided does not use type hints, "
                              "which we use when choosing actions."))

        return cls(
            name=fn.__name__,
            fn=fn,
            # list of classes, one for each input arg
            arg_types=list(types.values())[0:-1],
            return_type=types["return"],
        )

    @classmethod
    def op_from_const(cls, const: Any):
        """
        Creates a ForwardOp for a constant value. This just
        makes a function which takes zero arguments, and has the output type of
        the value type given.
        """
        return cls(
            name=str(const),
            fn=lambda: const,  # A function with zero arguments
            arg_types=[],
            return_type=type(const),
        )

    def evaluate(self, args: List):
        if len(args) != len(self.arg_types):
            raise ValueError("too many arguments given")

        # TODO: Potentially typecheck
        # If we want to typecheck arguments, we need to do something a bit more
        # complicated than the code below.
        # Like using the typeguard library.
        # for i, (arg, target_type) in enumerate(zip(arguments, self.arg_types)):
        #     if type(arg) != self.arg_types:
        #         raise TypeError(
        #             (f"Expected type {target_type} but got"
        #              f"{type(arg)} for argument {i} of {self.op_name}"))

        return self.fn(*args)


class InvertibleOp(ForwardOp):
    """
    This has all the same attributes as a ForwardOp, with the addition of an
    inverse_op function which returns a tuple of input arguments for the
    function. For now, let's assume inverse ops are one-to-one.
    """
    def __init__(
        self,
        name: str,
        fn: Callable,
        arg_types: List,
        return_type,
        inverse_fn: Callable,
    ):
        assert len(arg_types) == 1, "Only one-to-one functions supported"

        super().__init__(
            name=name,
            fn=fn,
            arg_types=arg_types,
            return_type=return_type,
        )
        self.inverse_fn = inverse_fn

    @classmethod
    def op_from_fn_pair(
        cls,
        fn: Callable,
        inverse_fn: Callable,
    ):
        """
        Creates a BidirOp for a given function. Infers types
        from type hints, so the op needs to be implemented with type hints.
        inverse_op returns a tuple of input arguments for the function, and
        doesn"t need any type hints.
        """
        types = typing.get_type_hints(fn)
        if len(types) == 0:
            raise ValueError(("Operation provided does not use type hints, "
                              "which we use when choosing actions."))

        return cls(
            name=fn.__name__,
            fn=fn,
            # list of classes, one for each input arg
            arg_types=list(types.values())[0:-1],
            return_type=types["return"],
            inverse_fn=inverse_fn,
        )

    def inverse_evaluate(self, arg: Any):
        # TODO: Potentially typecheck
        # if type(output) != self.return_type:
        #     raise TypeError((f"Expected type {self.return_type} but got "
        #                      f"{type(output)} for output"))

        return self.inverse_fn(arg)


_FUNCTIONS: List[Callable] = [
    F.area,
    F.color_i_to_j,
    F.color_in,
    F.columns,
    F.crop,
    F.deflate,
    F.empty_grid,
    F.filter_color,
    F.get_color,
    F.hflip,
    F.hstack_pair,
    F.hstack,
    F.inflate,
    F.kronecker,
    F.overlay_pair,
    F.overlay,
    F.rotate_ccw,
    F.rotate_cw,
    F.rows,
    F.set_bg,
    F.size,
    F.top_half,
    F.unset_bg,
    F.vflip,
    F.vstack_pair,
    F.vstack,
]

FORWARD_FUNCTION_OPS = [ForwardOp.op_from_fn(fn) for fn in _FUNCTIONS]

COLOR_OPS = [ForwardOp.op_from_const(c) for c in COLORS.ALL_COLORS]

BOOL_OPS = [ForwardOp.op_from_const(b) for b in [True, False]]

# stick to small ints for now?
INT_OPS = [ForwardOp.op_from_const(i) for i in range(3)]

all_forward_ops = FORWARD_FUNCTION_OPS + COLOR_OPS + BOOL_OPS + INT_OPS

# sticking to one-to-one functions for now.
# TODO: Should we move these defs into bidir.primitives.functions?
_FUNCTION_INVERSE_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.rotate_ccw, F.rotate_cw),
    (F.rotate_cw, F.rotate_ccw),
    (F.vflip, F.vflip),
    (F.hflip, F.hflip),
    (F.rows, F.vstack),
    (F.columns, F.hstack),
]

ALL_BIDIR_OPS = [
    InvertibleOp.op_from_fn_pair(op, inverse)
    for op, inverse in _FUNCTION_INVERSE_PAIRS
]
