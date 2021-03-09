from typing import List, Sequence, Tuple
import unittest

import networkx as nx
import numpy as np

import bidir.primitives.functions as F
from bidir.primitives.types import Grid
from bidir.task_utils import Task, twenty_four_task, arc_task
from bidir.utils import SynthError

from rl.agent import ProgrammableAgent, ProgrammableAgent2
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from rl.environment import SynthEnv, SynthEnvAction
from rl.ops.operations import ForwardOp, Op
import rl.ops.twenty_four_ops


def make_24_program(
        ops: Sequence[Op],
        commands: List[Tuple[str, Tuple[int,
                                        int]]]) -> List[SynthEnvAction]:
    op_dict = {op.name: op for op in ops}

    return [
        SynthEnvAction(op_dict[s], [ValueNode(
            (a1, )), ValueNode((a2, ))]) for (s, (a1, a2)) in commands
    ]


class ProgramSearchGraphTests(unittest.TestCase):

    def test_actions_in_program(self):
        task = twenty_four_task((1, ), 24)
        env = SynthEnv(task=task)

        ops = rl.ops.twenty_four_ops.ALL_OPS

        commands = [
            ('add', (1, 1)),  # 1 + 1 = 2
            ('add', (2, 2)),  # 2 + 2 = 4
            ('add', (2, 4)),  # 2 + 4 = 6
            ('mul', (4, 6)),  # 4 * 6 = 24
        ]

        program = make_24_program(ops, commands)
        agent = ProgrammableAgent(program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            _, _, done, _ = env.step(action)

        self.assertTrue(done)
        self.assertEqual(env.psg.actions_in_program(), {0, 1, 2, 3})

    def test_actions_in_program2(self):
        task = arc_task(30)
        env = SynthEnv(task=task)

        ops = rl.ops.arc_ops.ALL_OPS
        op_dict = {op.name: op for op in ops}

        def f(string):
            return op_dict[string]

        program = [
            (f('Color.BLACK'), (0, )),  # 2
            (f('Color.RED'), (0, )),  # 3
            (f('set_bg'), (0, 2)),  # 4
            (f('crop'), (4, )),  # 5
            (f('unset_bg'), (5, 2)),  # 6
        ]

        agent = ProgrammableAgent2(program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            _, _, done, _ = env.step(action)

        self.assertTrue(done)
        self.assertEqual(env.psg.actions_in_program(), {0, 2, 3, 4})

    def test_actions_in_program3(self):
        task = twenty_four_task((3, ), 9)
        env = SynthEnv(task=task)

        ops = rl.ops.twenty_four_ops.ALL_OPS

        commands = [
            ('add', (3, 3)),  # 3 + 3 = 6
            ('mul', (3, 3)),  # 3 * 3 = 9
        ]

        program = make_24_program(ops, commands)
        agent = ProgrammableAgent(program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            _, _, done, _ = env.step(action)

        self.assertTrue(done)
        self.assertEqual(env.psg.actions_in_program(), {1})

    def test_repeated_forward_op2(self):
        task = twenty_four_task((2, 3, 4, 7), 24)
        SYNTH_ERROR_PENALTY = -100
        env = SynthEnv(task=task, synth_error_penalty=SYNTH_ERROR_PENALTY)

        ops = rl.ops.twenty_four_ops.ALL_OPS

        commands = [
            ('sub', (7, 3)),  # 7 - 3 = 4
        ]

        program = make_24_program(ops, commands)
        agent = ProgrammableAgent(program)

        action = agent.choose_action(env.observation())
        _, reward, _, _ = env.step(action)
        self.assertEqual(reward, SYNTH_ERROR_PENALTY, "Need synth error")
        env.observation().psg.check_invariants()

    def test_repeated_forward_op(self):
        task = twenty_four_task((2, 2, 3, 1), 24)
        SYNTH_ERROR_PENALTY = -100
        env = SynthEnv(task=task, synth_error_penalty=SYNTH_ERROR_PENALTY)

        ops = rl.ops.twenty_four_ops.ALL_OPS

        commands = [
            ('mul', (2, 2)),  # 2 * 2 = 4
            ('add', (3, 1)),  # 3 + 1 = 4
        ]

        program = make_24_program(ops, commands)
        agent = ProgrammableAgent(program)

        action = agent.choose_action(env.observation())
        env.step(action)
        env.observation().psg.check_invariants()
        action = agent.choose_action(env.observation())

        _, reward, _, _ = env.step(action)
        self.assertEqual(reward, SYNTH_ERROR_PENALTY, "Need synth error")
        env.observation().psg.check_invariants()

    def test_repeated_inverse_op(self):
        task = twenty_four_task((2, 3, 9), 24)
        SYNTH_ERROR_PENALTY = -100
        env = SynthEnv(task=task, synth_error_penalty=SYNTH_ERROR_PENALTY)

        ops = rl.ops.twenty_four_ops.ALL_OPS

        commands = [
            ('mul_cond_inv', (3, 0)),  # 24 = 2 * ?12
            ('mul_cond_inv', (3, 0)),  # repeat
        ]

        program = make_24_program(ops, commands)
        agent = ProgrammableAgent(program)

        action = agent.choose_action(env.observation())
        env.step(action)
        env.observation().psg.check_invariants()
        action = agent.choose_action(env.observation())

        _, reward, _, _ = env.step(action)
        self.assertEqual(reward, SYNTH_ERROR_PENALTY, "Need synth error")

    def test_node_removal(self):
        task = twenty_four_task((2, 3, 9), 24)
        env = SynthEnv(task=task)

        ops = rl.ops.twenty_four_ops.ALL_OPS

        commands = [
            ('mul_cond_inv', (24, 2)),  # 24 = 2 * ?12
            ('mul_cond_inv', (12, 3)),  # 12 = 3 * ?4
            ('add', (9, 3)),  # 9 + 3 = 12
        ]

        program = make_24_program(ops, commands)
        agent = ProgrammableAgent(program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            env.step(action)
            env.observation().psg.check_invariants()

        old_psg = env.psg

        task = twenty_four_task((2, 3, 9), 24)
        env = SynthEnv(task=task)

        commands2 = [
            ('mul_cond_inv', (24, 2)),  # 24 = 2 * ?12
            ('add', (9, 3)),  # 9 + 3 = 12
        ]

        program2 = make_24_program(ops, commands2)
        agent = ProgrammableAgent(program2)

        for i in range(len(commands2)):
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
        task = Task((start_grids, ), end_grids)
        psg = ProgramSearchGraph(task)

        op1 = ForwardOp(F.rotate_ccw)
        op1.apply_op(psg, (psg.get_value_nodes()[0], ))
        op1.apply_op(psg, (psg.get_value_nodes()[2], ))
        op1.apply_op(psg, (psg.get_value_nodes()[3], ))
        try:
            op1.apply_op(psg, (psg.get_value_nodes()[4], ))
        except SynthError:
            pass
        else:
            self.assertFalse(True, 'Should cause error')

        self.assertTrue(nx.algorithms.dag.is_directed_acyclic_graph(psg.graph))
        psg.check_invariants()
