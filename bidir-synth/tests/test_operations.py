import unittest
import networkx as nx
import numpy as np

import bidir.primitives.functions as F
from bidir.primitives.types import Grid
from rl.program_search_graph import ProgramSearchGraph
from rl.operations import ForwardOp


class OperationTests(unittest.TestCase):
    def test_value_cycle(self):
        start_grids = (
            Grid(np.array([[0, 0], [1, 1]])),
            Grid(np.array([[2, 2], [2, 2]])),
        )

        end_grids = (
            Grid(np.array([[0, 0], [0, 0]])),
            Grid(np.array([[0, 0], [0, 0]])),
        )
        psg = ProgramSearchGraph((start_grids, ), end_grids)

        op1 = ForwardOp(F.rotate_ccw)
        op1.apply_op(psg, (psg.get_value_nodes()[0], ))
        op1.apply_op(psg, (psg.get_value_nodes()[2], ))
        op1.apply_op(psg, (psg.get_value_nodes()[3], ))
        op1.apply_op(psg, (psg.get_value_nodes()[4], ))

        self.assertTrue(nx.algorithms.dag.is_directed_acyclic_graph(psg.graph))
        psg.check_invariants()
