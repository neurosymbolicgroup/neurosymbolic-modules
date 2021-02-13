from typing import List, Tuple
import unittest

import networkx as nx
import numpy as np

import bidir.primitives.functions as F
from bidir.primitives.types import Grid
from rl.program_search_graph import ProgramSearchGraph
from rl.agent import ProgrammableAgent
from rl.environment import SynthEnv
from rl.ops.operations import ForwardOp
import rl.ops.twenty_four_ops


class ProgramSearchGraphTests(unittest.TestCase):
    def test_node_removal(self):
        train_exs = (((2, 3, 9), 24), )
        env = SynthEnv(train_exs, tuple(), rl.ops.twenty_four_ops.ALL_OPS)

        program: List[Tuple[str, Tuple[int, ...]]] = [
            ('mul_cond_inv', (3, 0)),  # 24 = 2 * ?12
            ('mul_cond_inv', (4, 1)),  # 12 = 3 * ?4
            ('add', (1, 2)),  # 9 + 3 = 12
        ]

        agent = ProgrammableAgent(env.ops, program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            env.step(action)
            env.observation().psg.check_invariants()

        old_psg = env.psg

        train_exs = (((2, 3, 9), 24), )
        env = SynthEnv(train_exs, tuple(), rl.ops.twenty_four_ops.ALL_OPS)

        program2: List[Tuple[str, Tuple[int, ...]]] = [
            ('mul_cond_inv', (3, 0)),  # 24 = 2 * ?12
            ('add', (1, 2)),  # 9 + 3 = 12
        ]

        agent = ProgrammableAgent(env.ops, program2)

        for i in range(len(program2)):
            action = agent.choose_action(env.observation())
            env.step(action)
            env.observation().psg.check_invariants()

        # the two graphs should be identical to each other
        psg = env.psg

        self.assertTrue(nx.is_isomorphic(psg.graph, old_psg.graph))

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
