from rl.functions import Function
from bidir.primitives.types import Grid
from typing import Tuple, List


class Program:
    def evaluate(self, input_grid: Grid):
        pass


class ProgFunction(Program):
    def __init__(self, fn: Function, args: List[Program]):
        self.fn = fn
        self.args = args
        assert len(self.args) == self.fn.arity
        # TODO: type check?

    def __str__(self):
        return f"{self.fn.name}({', '.join(str(a) for a in self.args)})"

    def __repr__(self):
        return str(self)

    def evaluate(self, input_grid: Grid):
        eval_args = [arg.evaluate(input_grid) for arg in self.args]
        out = self.fn.fn(*eval_args)
        return out


class ProgConstant(Program):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)

    def evaluate(self, input_grid: Grid):
        return self.value


class ProgInputGrids(Program):
    def __init__(self):
        pass

    def __str__(self):
        return "input"

    def __repr__(self):
        return str(self)

    def evaluate(self, input_grid: Grid):
        return input_grid


def eval_program_on_grids(program: Program, input_grids: Tuple[Grid]) -> Tuple:
    return tuple(program.evaluate(grid) for grid in input_grids)
