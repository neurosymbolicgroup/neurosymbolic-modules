from rl.operations import Op, ForwardOp, CondInverseOp
from typing import Callable, Tuple, Optional, List, Dict

MIN = 0
MAX = 100


class TwentyFourError(TypeError):
    pass


def bound_ints(fn):
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, float) and not out.is_integer():
            raise TwentyFourError
        if out < MIN or out > MAX:
            raise TwentyFourError
        return out

    return wrapper


# @bound_ints
def add(a: int, b: int) -> int:
    return a + b


# @bound_ints
def subtract(a: int, b: int) -> int:
    return a - b


# @bound_ints
def multiply(a: int, b: int) -> int:
    return a * b


# @bound_ints
def divide(a: int, b: int) -> int:
    if a % b != 0:
        raise TwentyFourError
    return a // b


def add_cond_inv(
    out: int, args: Tuple[Optional[int], Optional[int]]
) -> Tuple[Optional[int], Optional[int]]:
    a, b = args
    if (a is None) == (b is None):
        raise TwentyFourError
    if a is None:
        assert b is not None
        return (out - b, b)
    else:
        assert b is None
        assert a is not None
        return (a, out - a)


def subtract_cond_inv(
    out: int, args: Tuple[Optional[int], Optional[int]]
) -> Tuple[Optional[int], Optional[int]]:
    a, b = args
    if (a is None) == (b is None):
        raise TwentyFourError
    if a is None:
        assert b is not None
        return (out + b, b)
    else:
        assert b is None
        assert a is not None
        return (a, a - out)


def multiply_cond_inv(
    out: int, args: Tuple[Optional[int], Optional[int]]
) -> Tuple[Optional[int], Optional[int]]:
    a, b = args
    if (a is None) == (b is None):
        raise TwentyFourError
    if a is None:
        assert b is not None
        if out % b != 0:
            raise TwentyFourError
        return (out // b, b)
    else:
        assert b is None
        assert a is not None
        if out % a != 0:
            raise TwentyFourError
        return (a, out // a)


def divide_cond_inv(
    out: int, args: Tuple[Optional[int], Optional[int]]
) -> Tuple[Optional[int], Optional[int]]:
    a, b = args
    if (a is None) == (b is None):
        raise TwentyFourError
    if a is None:
        assert b is not None
        return (out * b, b)
    else:
        assert b is None
        assert a is not None
        return (a, out * a)


FUNCTIONS: List[Callable] = [add, subtract, multiply, divide]

FORWARD_OPS = [ForwardOp(fn) for fn in FUNCTIONS]

_FUNCTION_COND_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (add, add_cond_inv),
    (subtract, subtract_cond_inv),
    (multiply, multiply_cond_inv),
    (divide, divide_cond_inv),
]

COND_INV_OPS = [
    CondInverseOp(fn, inverse_fn)
    for (fn, inverse_fn) in _FUNCTION_COND_INV_PAIRS
]

ALL_OPS = FORWARD_OPS + COND_INV_OPS  # type: ignore

assert len(set(op.name
               for op in ALL_OPS)) == len(ALL_OPS), ("duplicate op name")

OP_DICT: Dict[str, Op] = {op.name: op for op in ALL_OPS}
