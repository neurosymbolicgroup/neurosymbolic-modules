from typing import Any


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
