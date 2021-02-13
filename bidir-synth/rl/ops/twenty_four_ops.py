from typing import Callable, Tuple, List, Dict

from bidir.utils import SynthError
from rl.ops.operations import Op, ForwardOp, CondInverseOp

MAX_INT = 100


def add(a: int, b: int) -> int:
    out = a + b
    if out < 0 or out > MAX_INT:
        raise SynthError
    return out


def sub(a: int, b: int) -> int:
    if b > a:
        raise SynthError
    out = a - b
    if out < 0 or out > MAX_INT:
        raise SynthError
    return out


def mul(a: int, b: int) -> int:
    out = a * b
    if out < 0 or out > MAX_INT:
        raise SynthError
    return out


def div(a: int, b: int) -> int:
    if b == 0 or a % b != 0:
        raise SynthError
    out = a // b
    if out < 0 or out > MAX_INT:
        raise SynthError
    return out


def add_cond_inv(out: int, arg: int) -> Tuple[int]:
    if arg > out:
        raise SynthError
    out = out - arg
    if out < 0 or out > MAX_INT:
        raise SynthError
    return (out, )


def sub_cond_inv1(out: int, arg: int) -> Tuple[int]:
    # out = arg - ?
    # ? = arg - out
    if arg < out:
        raise SynthError
    out = arg - out
    if out < 0 or out > MAX_INT:
        raise SynthError
    return (out, )


def sub_cond_inv2(out: int, arg: int) -> Tuple[int]:
    # out = ? - arg
    # ? = out + arg
    out = out + arg
    if out < 0 or out > MAX_INT:
        raise SynthError
    return (out, )


def mul_cond_inv(out: int, arg: int) -> Tuple[int]:
    # out = arg * ?
    # ? = out / arg
    if arg == 0 or out % arg != 0:
        raise SynthError
    out = out // arg
    if out < 0 or out > MAX_INT:
        raise SynthError
    return (out, )


def div_cond_inv1(out: int, arg: int) -> Tuple[int]:
    # out = arg / ?
    # ? = arg / out
    if arg == 0 or arg % out != 0:
        raise SynthError
    out = arg // out
    if out < 0 or out > MAX_INT:
        raise SynthError
    return (out, )


def div_cond_inv2(out: int, arg: int) -> Tuple[int]:
    # out = ? / arg
    # ? = out * arg
    out = out * arg
    if out < 0 or out > MAX_INT:
        raise SynthError
    return (out, )


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

OP_DICT = {op.name: op for op in ALL_OPS}
