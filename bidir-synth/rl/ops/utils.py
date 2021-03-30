from typing import Callable, Dict, List, Sequence
from rl.ops.operations import Op, InverseOp, CondInverseOp, ForwardOp


def tuple_return(f: Callable):
    """
    Inverse functions should take in the output and return a tuple of input
    arguments.
    This function is useful if the inverse function is already implemented as a
    forward function, and we simply want it to return a tuple with a single
    element instead of a single value.
    For example, the proper inverse of rotate_cw is tuple_return(rotate_ccw).
    """
    return lambda x: (f(x), )


def fw_dict(ops: Sequence[Op]) -> Dict[str, ForwardOp]:
    return {op.forward_fn.name: op for op in ops if isinstance(op, ForwardOp)}


def inv_dict(ops: Sequence[Op]) -> Dict[str, InverseOp]:
    return {op.forward_fn.name: op for op in ops if isinstance(op, InverseOp)}


def cond_inv_dict(ops: Sequence[Op]) -> Dict[str, List[CondInverseOp]]:
    d: Dict[str, List[CondInverseOp]] = {}
    for op in ops:
        if isinstance(op, CondInverseOp):
            if op.forward_fn.name in d:
                d[op.forward_fn.name].append(op)
            else:
                d[op.forward_fn.name] = [op]
    return d
