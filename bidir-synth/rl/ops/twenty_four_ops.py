from typing import Callable, Tuple, List

from rl.ops.utils import tuple_return
from bidir.utils import SynthError
from bidir.primitives.functions import Function, make_function
from rl.ops.operations import ForwardOp, CondInverseOp, Op, InverseOp

MAX_INT = 100


def add(a: int, b: int) -> int:
    out = a + b
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return out


def sub(a: int, b: int) -> int:
    if b > a:
        raise SynthError("24")
    out = a - b
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return out


def mul(a: int, b: int) -> int:
    out = a * b
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return out


def div(a: int, b: int) -> int:
    if b == 0 or a % b != 0:
        raise SynthError("24")
    out = a // b
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return out


def add_cond_inv(out: int, arg: int) -> Tuple[int]:
    if arg > out:
        raise SynthError("24")
    out = out - arg
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return (out, )


def sub_cond_inv1(out: int, arg: int) -> Tuple[int]:
    # out = arg - ?
    # ? = arg - out
    if arg < out:
        raise SynthError("24")
    out = arg - out
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return (out, )


def sub_cond_inv2(out: int, arg: int) -> Tuple[int]:
    # out = ? - arg
    # ? = out + arg
    out = out + arg
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return (out, )


def mul_cond_inv(out: int, arg: int) -> Tuple[int]:
    # out = arg * ?
    # ? = out / arg
    if arg == 0 or out % arg != 0:
        raise SynthError("24")
    out = out // arg
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return (out, )


def div_cond_inv1(out: int, arg: int) -> Tuple[int]:
    # out = arg / ?
    # ? = arg / out
    if arg == 0 or arg % out != 0:
        raise SynthError("24")
    out = arg // out
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return (out, )


def div_cond_inv2(out: int, arg: int) -> Tuple[int]:
    # out = ? / arg
    # ? = out * arg
    out = out * arg
    if out < 0 or out > MAX_INT:
        raise SynthError("24")
    return (out, )


SPECIAL_OPS = {
    'a - b': lambda a, b: a - b,
    '2a - b': lambda a, b: 2 * a - b,
    '3a - b': lambda a, b: 3 * a - b,
    # hack: -1 will trigger synth error
    'a / b': lambda a, b: a // b if b != 0 else -1,
    'a - 2b': lambda a, b: a - 2 * b,
    'a - 3b': lambda a, b: a - 3 * b,
    '3a - 2b': lambda a, b: 3 * a - 2 * b,
    '2a - 3b': lambda a, b: 2 * a - 3 * b,
    'a + 2b': lambda a, b: a + 2 * b,
    'a + 3b': lambda a, b: a + 3 * b,
    'a * (b + 1)': lambda a, b: a * (b + 1),
    'a * (b + 2)': lambda a, b: a * (b + 2),
    'a * (b + 3)': lambda a, b: a * (b + 3),
    '(a + 1) * (b + 2)': lambda a, b: (a + 1) * (b + 2),
    '(a + 1) * (b + 3)': lambda a, b: (a + 1) * (b + 3),
}


def wrap_special_fn(fn: Callable) -> Callable:
    """
    Gives the fn some type hints and bound checking.
    """
    def f(a: int, b: int) -> int:
        out = fn(a, b)
        if not float(out).is_integer() or out < 0 or out > MAX_INT:
            raise SynthError("24")
        return out

    return f

def minus1(x: int) -> int:
    return x - 1

def plus1(x: int) -> int:
    return x + 1


MINUS1_OP = ForwardOp(minus1)
MINUS1_INV_OP = InverseOp(minus1, tuple_return(plus1))

SPECIAL_FNS = [wrap_special_fn(op_fn) for op_fn in SPECIAL_OPS.values()]

SPECIAL_FORWARD_OPS = [ForwardOp(fn) for fn in SPECIAL_FNS]

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
