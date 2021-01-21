class ArcError(Exception):
    pass

def soft_assert(condition: bool):
    """
    For sanity checking. The assertion fails silently and
    enumeration will continue, but whatever program caused the assertion is
    immediately discarded as nonviable. This is useful for checking correct
    inputs, not creating a massive grid that uses up memory by kronecker super
    large grids, and so on.
    """
    if not condition:
        raise ArcError
