import typing
from typing import Any, Callable, List, Dict


class Function:
    def __init__(
        self,
        name: str,
        fn: Callable,
        arg_types: List[type],
        return_type: type,
    ):
        self.name = name
        self.fn = fn
        self.arg_types = arg_types
        self.arity: int = len(self.arg_types)
        self.return_type: type = return_type

    @classmethod
    def from_typed_fn(cls, fn: Callable):
        """
        Creates a Function for the given function. Infers types from type hints,
        so the op needs to be implemented with type hints.
        """
        types: Dict[str, type] = typing.get_type_hints(fn)
        if len(types) == 0:
            raise ValueError(("Operation provided does not use type hints, "
                              "which we use when choosing ops."))

        return cls(
            name=fn.__name__,
            fn=fn,
            # list of classes, one for each input arg. skip last type (return)
            arg_types=list(types.values())[0:-1],
            return_type=types["return"],
        )

    def __str__(self):
        return self.name


class Op:
    def __init__(self, fn: Function, inverse_fn: Callable, tp: str):
        self.fn = fn
        # if forwards, not needed
        self.inverse_fn = inverse_fn
        # 'forward', 'inverse', or 'cond inverse'
        self.tp = tp


def forward_op(fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=None, tp='forward')


def constant_op(cons: Any, name: str = None):
    if name is None:
        name = str(cons)

    fn = Function(
        name=str(cons),
        fn=lambda: cons,
        arg_types=[],
        return_type=type(cons),
    )
    return Op(fn=fn, inverse_fn=None, tp='forward')


def inverse_op(fn: Callable, inverse_fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=inverse_fn, tp='inverse')


def cond_inverse_op(fn: Callable, inverse_fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=inverse_fn, tp='cond inverse')
