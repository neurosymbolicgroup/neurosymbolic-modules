import unittest
from typing import List, Tuple, Union, Optional
from rl.agent import ProgrammableAgent
from rl.create_ops import OP_DICT
from rl.environment import ArcEnvironment
from bidir.task_utils import get_task_examples
from rl.program import eval_program_on_grids


class TestProgramAgent(unittest.TestCase):
    def check_program_on_task(
        self,
        task_num: int,
        program: List[Tuple[Union[str, Optional[int]], ...]],
        train: bool = True,
    ):

        train_exs, test_exs = get_task_examples(task_num, train=train)
        train_inputs, train_outputs = zip(*train_exs)
        test_inputs, test_outputs = zip(*test_exs)

        env = ArcEnvironment(train_exs, test_exs, max_actions=-1)
        # each subprogram generates one solution for the task
        # the subprograms are executed sequentially, without resetting the env in
        # between
        for subprogram in program:
            agent = ProgrammableAgent(OP_DICT, subprogram)

            while not agent.done:
                action = agent.choose_action(env.state)
                state, reward, done = env.step(action)
                if not agent.done:
                    self.assertFalse(done)

            self.assertTrue(done)
            prog = env.state.get_program()
            print(f'Program generated from agent behavior: {prog}')
            self.assertEqual(eval_program_on_grids(prog, train_inputs),
                             train_outputs)
            self.assertEqual(eval_program_on_grids(prog, test_inputs),
                             test_outputs)


    def get_train_program(
        self,
        task_num: int,
    ) -> Union[str, List[List[Tuple]]]:
        # yapf: disable  # don't want program lines being joined
        if task_num == 56:
            program = [
                ('Black', 0),
                ('set_bg', 0, 2),
                ('crop', 3),
                ('1', 0),
                ('2', 0),
                ('block', 5, 6, 2),
                ('kronecker', 7, 4),
                ('unset_bg', 8, 2),
            ]
            return [program]
        elif task_num == 82:
            # Simon's pretty proud of this one :)
            program = [
                ('hflip', 0),
                ('hstack_pair', 0, 2),
                ('vstack_pair_cond_inv', 1, 3, None),
                ('vflip', 0),
                ('hstack_pair_cond_inv', 4, 5, None),
                ('vflip_inv', 6)
            ]
            return [program]
        elif task_num == 86:
            program = [
                ('rotate_cw_inv', 1),
                ('rotate_cw_inv', 2),
            ]
            return [program]
        elif task_num == 112:
            program = [
                [   ('Black', 0),
                    ('set_bg', 0, 2),
                    ('vflip', 3),
                    ('overlay_pair', 3, 4),
                    ('unset_bg', 5, 2) ],
                [   ('Grey', 0),
                    ('Red', 0),
                    ('Yellow', 0),
                    ('color_i_to_j', 6, 7, 8),],
                [   ('color_i_to_j', 6, 7, 9),]
            ]
            return program
        elif task_num == 115:
            program = [
                ('vflip', 0),
                ('vstack_pair_cond_inv', 1, 2, None),
            ]
            return [program]
        else:
            return "No program"
        # yapf: enable

    def test_on_train_tasks(self):
        total_solved = 0

        for task_num in range(400):
            program = self.get_train_program(task_num)
            if isinstance(program, str):
                continue

            with self.subTest(task_num=task_num):
                self.assertNotEqual(program, None,
                                    (f"program for {task_num} is None. "
                                     f"Did you forget to 'return [program]'?"))
                self.check_program_on_task(task_num, program)
                total_solved += 1

        print(f"\nSolved {total_solved} ARC tasks with RL programmable agent,",
              "and confirmed each generated AST evaluates correctly too!")
