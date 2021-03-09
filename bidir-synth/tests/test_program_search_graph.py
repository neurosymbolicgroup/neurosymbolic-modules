from typing import List
import unittest

import networkx as nx
import numpy as np

import bidir.primitives.functions as F
from bidir.primitives.types import Grid
from bidir.task_utils import Task, twenty_four_task, arc_task
from bidir.utils import SynthError

from rl.agent import ProgrammableAgent
from rl.program_search_graph import ProgramSearchGraph
from rl.environment import SynthEnv, SynthEnvAction
from rl.ops.operations import ForwardOp
import rl.ops.twenty_four_ops


class ProgramSearchGraphTests(unittest.TestCase):
    def test_actions_in_program(self):
        task = twenty_four_task((1, ), 24)
        env = SynthEnv(task=task, ops=rl.ops.twenty_four_ops.ALL_OPS)

        op_names = list(rl.ops.twenty_four_ops.OP_DICT.keys())

        def f(string):
            return op_names.index(string)

        program: List[SynthEnvAction] = [
            SynthEnvAction(f('add'), (0, 0)),  # 1 + 1 = 2
            SynthEnvAction(f('add'), (2, 2)),  # 2 + 2 = 4
            SynthEnvAction(f('add'), (2, 3)),  # 2 + 3 = 6
            SynthEnvAction(f('mul'), (3, 4)),  # 4 * 6 = 24
        ]

        agent = ProgrammableAgent(env.ops, program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            _, _, done, _ = env.step(action)

        self.assertTrue(done)
        self.assertEqual(env.psg.actions_in_program(), {0, 1, 2, 3})

    def test_actions_in_program2(self):
        task = arc_task(30)
        env = SynthEnv(task=task, ops=rl.ops.arc_ops.ALL_OPS)

        op_names = list(rl.ops.arc_ops.OP_DICT.keys())

        def f(string):
            return op_names.index(string)

        program: List[SynthEnvAction] = [
            SynthEnvAction(f('Color.BLACK'), (0, )),  # 2
            SynthEnvAction(f('Color.RED'), (0, )),  # 3
            SynthEnvAction(f('set_bg'), (0, 2)),  # 4
            SynthEnvAction(f('crop'), (4, )),  # 5
            SynthEnvAction(f('unset_bg'), (5, 2)),  # 6
        ]

        agent = ProgrammableAgent(env.ops, program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            _, _, done, _ = env.step(action)

        self.assertTrue(done)
        self.assertEqual(env.psg.actions_in_program(), {0, 2, 3, 4})

    def test_actions_in_program3(self):
        task = twenty_four_task((3, ), 9)
        env = SynthEnv(task=task, ops=rl.ops.twenty_four_ops.ALL_OPS)

        op_names = list(rl.ops.twenty_four_ops.OP_DICT.keys())

        def f(string):
            return op_names.index(string)

        program: List[SynthEnvAction] = [
            SynthEnvAction(f('add'), (0, 0)),  # 3 + 3 = 6
            SynthEnvAction(f('mul'), (0, 0)),  # 3 * 3 = 9
        ]

        agent = ProgrammableAgent(env.ops, program)
        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            _, _, done, _ = env.step(action)

        self.assertTrue(done)
        self.assertEqual(env.psg.actions_in_program(), {1})

    def test_repeated_forward_op2(self):
        task = twenty_four_task((2, 3, 4, 7), 24)
        SYNTH_ERROR_PENALTY = -100
        env = SynthEnv(task=task,
                       ops=rl.ops.twenty_four_ops.ALL_OPS,
                       synth_error_penalty=SYNTH_ERROR_PENALTY)

        ops = list(rl.ops.twenty_four_ops.OP_DICT.keys())
        program: List[SynthEnvAction] = [
            SynthEnvAction(ops.index('sub'), (3, 1)),  # 7 - 3 = 4
        ]

        agent = ProgrammableAgent(env.ops, program)

        action = agent.choose_action(env.observation())
        _, reward, _, _ = env.step(action)
        self.assertEqual(reward, SYNTH_ERROR_PENALTY, "Need synth error")
        env.observation().psg.check_invariants()

    def test_repeated_forward_op(self):
        task = twenty_four_task((2, 2, 3, 1), 24)
        SYNTH_ERROR_PENALTY = -100
        env = SynthEnv(task=task,
                       ops=rl.ops.twenty_four_ops.ALL_OPS,
                       synth_error_penalty=SYNTH_ERROR_PENALTY)

        ops = list(rl.ops.twenty_four_ops.OP_DICT.keys())
        program: List[SynthEnvAction] = [
            SynthEnvAction(ops.index('mul'), (0, 1)),
            SynthEnvAction(ops.index('add'), (2, 3)),
        ]

        agent = ProgrammableAgent(env.ops, program)

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
        env = SynthEnv(task=task,
                       ops=rl.ops.twenty_four_ops.ALL_OPS,
                       synth_error_penalty=SYNTH_ERROR_PENALTY)

        ops = list(rl.ops.twenty_four_ops.OP_DICT.keys())
        program: List[SynthEnvAction] = [
            SynthEnvAction(ops.index('mul_cond_inv'), (3, 0)),  # 24 = 2 * ?12
            SynthEnvAction(ops.index('mul_cond_inv'), (3, 0)),  # repeat
        ]

        agent = ProgrammableAgent(env.ops, program)

        action = agent.choose_action(env.observation())
        env.step(action)
        env.observation().psg.check_invariants()
        action = agent.choose_action(env.observation())

        _, reward, _, _ = env.step(action)
        self.assertEqual(reward, SYNTH_ERROR_PENALTY, "Need synth error")

    def test_node_removal(self):
        task = twenty_four_task((2, 3, 9), 24)
        env = SynthEnv(task=task, ops=rl.ops.twenty_four_ops.ALL_OPS)

        ops = list(rl.ops.twenty_four_ops.OP_DICT.keys())
        program: List[SynthEnvAction] = [
            SynthEnvAction(ops.index('mul_cond_inv'), (3, 0)),  # 24 = 2 * ?12
            SynthEnvAction(ops.index('mul_cond_inv'), (4, 1)),  # 12 = 3 * ?4
            SynthEnvAction(ops.index('add'), (1, 2)),  # 9 + 3 = 12
        ]

        agent = ProgrammableAgent(env.ops, program)

        for _ in range(len(program)):
            action = agent.choose_action(env.observation())
            env.step(action)
            env.observation().psg.check_invariants()

        old_psg = env.psg

        task = twenty_four_task((2, 3, 9), 24)
        env = SynthEnv(task=task, ops=rl.ops.twenty_four_ops.ALL_OPS)

        program2: List[SynthEnvAction] = [
            SynthEnvAction(ops.index('mul_cond_inv'), (3, 0)),  # 24 = 2 * ?12
            SynthEnvAction(ops.index('add'), (1, 2)),  # 9 + 3 = 12
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
