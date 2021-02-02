from rl.operations import Op, ForwardOp, CondInverseOp
from typing import Callable, Tuple, Optional, List, Dict

MAX = 100


class TwentyFourError(TypeError):
    pass


def bound_ints(fn: Callable[[int, int], int]) -> Callable[[int, int], int]:
    def wrapper(a: int, b: int) -> int:
        out = fn(a, b)
        if out < 0 or out > MAX:
            raise TwentyFourError
        return out

    return wrapper


# decorators currently not supported since it messes up type hints
# @bound_ints
def add(a: int, b: int) -> int:
    return a + b


# @bound_ints
def sub(a: int, b: int) -> int:
    return a - b


# @bound_ints
def mul(a: int, b: int) -> int:
    return a * b


# @bound_ints
def div(a: int, b: int) -> int:
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


def sub_cond_inv(
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


def mul_cond_inv(
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


def div_cond_inv(
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


FUNCTIONS: List[Callable] = [add, sub, mul, div]

FORWARD_OPS = [ForwardOp(fn) for fn in FUNCTIONS]

_FUNCTION_COND_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (add, add_cond_inv),
    (sub, sub_cond_inv),
    (mul, mul_cond_inv),
    (div, div_cond_inv),
]

COND_INV_OPS = [
    CondInverseOp(fn, inverse_fn)
    for (fn, inverse_fn) in _FUNCTION_COND_INV_PAIRS
]

ALL_OPS = FORWARD_OPS + COND_INV_OPS  # type: ignore

assert len(set(op.name for op in ALL_OPS)) == len(ALL_OPS), (
    f"duplicate op name: {[op.name for op in ALL_OPS]}")

OP_DICT: Dict[str, Op] = {op.name: op for op in ALL_OPS}
