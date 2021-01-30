from bidir.primitives.functions import Function
from bidir.primitives.types import Grid
from typing import Any, List


class Program:
    def evaluate(self, input_grid: Grid) -> Any:
        raise NotImplementedError


class ProgFunction(Program):
    def __init__(self, fn: Function, args: List[Program]):
        self.fn = fn
        self.args = args
        assert len(self.args) == self.fn.arity
        # TODO: type check?

    def __str__(self):
        return f"{self.fn.name}({', '.join(str(a) for a in self.args)})"

    def __repr__(self):
        return f"FUNCTION: {self}"

    def evaluate(self, input_grid: Grid) -> Any:
        arg_vals = [arg.evaluate(input_grid) for arg in self.args]
        return self.fn.fn(*arg_vals)


class ProgConstant(Program):
    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"CONST: {self}"

    def evaluate(self, input_grid: Grid) -> Any:
        return self.value


class ProgInputGrid(Program):
    def __str__(self):
        return "$INPUT"

    def __repr__(self):
        return str(self)

    def evaluate(self, input_grid: Grid) -> Grid:
        return input_grid
