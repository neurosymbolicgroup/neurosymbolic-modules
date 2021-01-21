from typing import Any, List, Set
from rl.operations import Function

"""
These are sketched classes/methods that I call in rl/operations.py, meant to be a
sort of interface between the underlying graph data structure and the RL
actions.

Feel free to change any of these in their names or arguments and such. If you
do, just change where they're called in rl/operations.py or let Simon know and
he'll do it.
"""

class ValueNode:
    """
    Value nodes are what we called "input ports" and "output ports".
    They have a value, and are either grounded or not grounded.
    """
    def __init__(self, value: Any, is_grounded: bool):
        self.value = value
        self.is_grounded = is_grounded


def get_value_nodes() -> List[ValueNode]
    """
    Returns the value nodes in the graph.
    This will be used by the RL agent to choose an action.
    """
    pass


def add_hyperedge(
    in_nodes: List[ValueNode],
    out_nodes: List[ValueNode],
    fn: Function
) -> None:
    """
    Adds the hyperedge to the data structure.
    This can be represented underneath however is most convenient.
    This method itself could even be changed, just go change where it's called
    """
    pass
