from typing import Callable, List, Tuple

from rl.ops.utils import tuple_return
from bidir.utils import SynthError
from bidir.task_utils import binary_task
from rl.ops.operations import ForwardOp, InverseOp
from rl.environment import SynthEnvAction
from rl.random_programs import ProgramSpec, ActionSpec

MAX_INT = 100

def plus_one(a: int) -> int:
    out = a + 1
    if out > MAX_INT:
        raise SynthError("binary")
    return out


def double(a: int) -> int:
    out = 2 * a
    if out > MAX_INT:
        raise SynthError("binary")
    return out


def plus_one_inv(a: int) -> int:
    out = a - 1
    if out <= 0:
        raise SynthError("binary")
    return out


def double_inv(a: int) -> int:
    if a % 2 != 0:
        raise SynthError("binary")
    out = a // 2
    return out


FUNCTIONS: List[Callable] = [plus_one, double]

FORWARD_OPS = [ForwardOp(fn, erase=True) for fn in FUNCTIONS]

INVERSE_FUNCTIONS: List[Callable] = [plus_one_inv, double_inv]

_FUNCTION_INV_PAIRS: List[Tuple[Callable, Callable]] = [
    (plus_one, tuple_return(plus_one_inv)),
    (double, tuple_return(double_inv)),
]

INVERSE_OPS = [
    InverseOp(fn, inverse_fn, erase=True)
    for (fn, inverse_fn) in _FUNCTION_INV_PAIRS
]

ALL_OPS = FORWARD_OPS + INVERSE_OPS  # type: ignore

assert len(set(op.name for op in ALL_OPS)) == len(ALL_OPS), (
    f"duplicate op name: {[op.name for op in ALL_OPS]}")

OP_DICT = {op.name: op for op in ALL_OPS}


def make_binary_program(target: int, forward_only: bool = True) -> ProgramSpec:
    if forward_only:
        return make_forward_binary_program(target)
    else:
        return make_bidir_binary_program(target)


def make_forward_binary_program(target: int) -> ProgramSpec:
    plus_one_op_idx = 0
    double_op_idx = 1

    def op_list(target: int) -> List[str]:
        assert target >= 2
        if target == 2:
            return []
        elif target % 2 == 0:
            return op_list(target // 2) + ['double']
        else:
            return op_list(target - 1) + ['plus_one']

    ops = op_list(target)
    # print(f"{target}: {len(ops}")

    program: List[SynthEnvAction] = []
    action_specs = []
    current_start = 2
    for op in ops:
        if op == 'double':
            op_idx = double_op_idx
        else:
            assert op == 'plus_one'
            op_idx = plus_one_op_idx

        arg_idx = 0  # always zero for supervised training.
        # if len(program) > 0:
            # arg_idx = 1  # newest node will be grounded
        action = SynthEnvAction(op_idx=op_idx, arg_idxs=(arg_idx, ))
        program.append(action)
        action_specs.append(ActionSpec(task=binary_task(target, start=current_start),
                                       action=action))

        if op == 'double':
            current_start *= 2
        else:
            assert op == 'plus_one'
            current_start += 1

    # task = binary_task(target)
    # assert rl_prog_solves(program=program, task=task, ops=FORWARD_OPS)
    return ProgramSpec(action_specs)


def program_length(target: int) -> int:
    if target == 2:
        return 0
    elif target % 2 == 0:
        return 1 + program_length(target // 2)
    else:
        return 1 + program_length(target - 1)


def make_bidir_binary_program(target: int, forward_only: bool = True) -> ProgramSpec:
    plus_one_inv_op_idx = 2
    double_inv_op_idx = 3

    def op_list(target: int) -> List[str]:
        assert target >= 2
        if target == 2:
            return []
        elif target % 2 == 0:
            return ['double_inv'] + op_list(target // 2)
        else:
            return ['plus_one_inv'] + op_list(target - 1)

    ops = op_list(target)
    # print(f"{target}: {ops}")

    program: List[SynthEnvAction] = []
    action_specs: List[ActionSpec] = []
    current_target = target

    for op in ops:
        if op == 'double_inv':
            op_idx = double_inv_op_idx
        else:
            assert op == 'plus_one_inv'
            op_idx = plus_one_inv_op_idx

        arg_idx = 1  # always the newest node
        action = SynthEnvAction(op_idx=op_idx, arg_idxs=(arg_idx, ))
        program.append(action)
        action_specs.append(ActionSpec(task=binary_task(current_target),
                                       action=action))

        if op == 'double_inv':
            current_target //= 2
        else:
            assert op == 'plus_one_inv'
            current_target -= 1

    # task = binary_task(target)
    # assert rl_prog_solves(program=program, task=task, ops=ALL_OPS)
    return ProgramSpec(action_specs)


def max_actions(max_target: int) -> int:
    return max([program_length(i) for i in range(3, max_target)])


if __name__ == '__main__':
    pass
    # for i in range(3, 100):
    #     prog = make_forward_binary_program(i)
    #     prog = make_bidir_binary_program(i)
