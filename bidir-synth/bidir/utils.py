class SynthError(Exception):
    """
    Use this for checking correct inputs, not creating a massive grid that uses
    up memory by kronecker super large grids, etc.
    """
    pass


def soft_assert(condition: bool):
    """
    Use this for checking correct inputs, not creating a massive grid that uses
    up memory by kronecker super large grids, etc.
    """
    if not condition:
        raise SynthError
