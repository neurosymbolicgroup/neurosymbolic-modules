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
    if b > a:
        raise TwentyFourError
    return a - b


# @bound_ints
def mul(a: int, b: int) -> int:
    return a * b


# @bound_ints
def div(a: int, b: int) -> int:
    if b == 0 or a % b != 0:
        raise TwentyFourError
    return a // b


def add_cond_inv(out: int, arg: int) -> Tuple[int]:
    if arg > out:
        raise TwentyFourError
    return (out - arg, )


def sub_cond_inv1(out: int, arg: int) -> Tuple[int]:
    # out = arg - ?
    # ? = arg - out
    if arg < out:
        raise TwentyFourError
    return (arg - out, )


def sub_cond_inv2(out: int, arg: int) -> Tuple[int]:
    # out = ? - arg
    # ? = out + arg
    return (out + arg, )


def mul_cond_inv(out: int, arg: int) -> Tuple[int]:
    # out = arg * ?
    # ? = out / arg
    if out % arg != 0:
        raise TwentyFourError
    return (out // arg, )


def div_cond_inv1(out: int, arg: int) -> Tuple[int]:
    # out = arg / ?
    # ? = arg / out
    if arg % out != 0:
        raise TwentyFourError
    return (arg // out, )


def div_cond_inv2(out: int, arg: int) -> Tuple[int]:
    # out = ? / arg
    # ? = out * arg
    return (out * arg, )


FUNCTIONS: List[Callable] = [add, sub, mul, div]

FORWARD_OPS = [ForwardOp(fn) for fn in FUNCTIONS]

_FUNCTION_COND_INV_PAIRS: List[Tuple[Callable, Callable, List[bool]]] = [
    (add, add_cond_inv, [True, False]),
    (sub, sub_cond_inv1, [True, False]),
    (sub, sub_cond_inv2, [False, True]),
    (mul, mul_cond_inv, [True, False]),
    (div, div_cond_inv1, [True, False]),
    (div, div_cond_inv2, [False, True]),
]

COND_INV_OPS = [
    CondInverseOp(fn, inverse_fn, expects_cond)
    for (fn, inverse_fn, expects_cond) in _FUNCTION_COND_INV_PAIRS
]

ALL_OPS = FORWARD_OPS + COND_INV_OPS  # type: ignore

assert len(set(op.name for op in ALL_OPS)) == len(ALL_OPS), (
    f"duplicate op name: {[op.name for op in ALL_OPS]}")

OP_DICT: Dict[str, Op] = {op.name: op for op in ALL_OPS}
