from typing import Callable, Dict, List
from rl.ops.operations import Op


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
