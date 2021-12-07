"""
single int as input, single int as target, operations just take you to a new
int.

for example, encode as binary, then ops are add1, add2, add4, etc.

Or maybe encode in base 2, and then ops are add1, times2, times10, etc.
And give a bound on total number of ops.
"""


from typing import Callable, List, Tuple

from rl.ops.utils import tuple_return
from bidir.utils import SynthError
from bidir.task_utils import binary_task
from rl.ops.operations import ForwardOp, InverseOp
from rl.environment import SynthEnvAction
from rl.random_programs import ProgramSpec, ActionSpec

MAX_INT = 10

def add1(a: int) -> int:
    out = a + 1
    if out > MAX_INT:
        raise SynthError("max int")
    return out

def add2(a: int) -> int:
    out = a + 2
    if out > MAX_INT:
        raise SynthError("max int")
    return out

def add4(a: int) -> int:
    out = a + 4
    if out > MAX_INT:
        raise SynthError("max int")
    return out

def add8(a: int) -> int:
    out = a + 8
    if out > MAX_INT:
        raise SynthError("max int")
    return out

def add16(a: int) -> int:
    out = a + 16
    if out > MAX_INT:
        raise SynthError("max int")
    return out

def add32(a: int) -> int:
    out = a + 32
    if out > MAX_INT:
        raise SynthError("max int")
    return out

def add64(a: int) -> int:
    out = a + 64
    if out > MAX_INT:
        raise SynthError("max int")
    return out

FUNCTIONS: List[Callable] = [add1, add2, add4, add8, add16, add32, add64]

FORWARD_OPS = [ForwardOp(fn, erase=False) for fn in FUNCTIONS]

ALL_OPS = FORWARD_OPS

assert len(set(op.name for op in ALL_OPS)) == len(ALL_OPS), (
    f"duplicate op name: {[op.name for op in ALL_OPS]}")

OP_DICT = {op.name: op for op in ALL_OPS}
