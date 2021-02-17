from typing import Any
import os


class SynthError(Exception):
    """
    Use this for checking correct inputs, not creating a massive grid that uses
    up memory by kronecker super large grids, etc.
    """


def soft_assert(condition: bool):
    """
    Use this for checking correct inputs, not creating a massive grid that uses
    up memory by kronecker super large grids, etc.
    """
    if not condition:
        raise SynthError


def assertEqual(a: Any, b: Any):
    assert a == b, f"expected {b} but got {a}"


def next_unused_path(path):
    last_dot = path.rindex('.')
    extension = path[last_dot:]
    file_name = path[:last_dot]

    i = 0
    while os.path.isfile(path):
        path = file_name + f"__{i}" + extension
        i += 1

    return path
