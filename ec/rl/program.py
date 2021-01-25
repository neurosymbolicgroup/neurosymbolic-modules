from rl.functions import Function
from bidir.primitives.types import Grid
from typing import Tuple, List, Any


class Program:
    def evaluate(self, num_examples: int) -> Tuple:
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

    def evaluate(self, num_examples: int) -> Tuple:
        # (num_args, num_examples)
        eval_args = [arg.evaluate(num_examples) for arg in self.args]
        num_examples = len(eval_args[0])
        out = []
        for i in range(num_examples):
            ex = self.fn.fn(*[arg[i] for arg in eval_args])
            out.append(ex)
        return tuple(out)


class ProgConstant(Program):
    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)

    def evaluate(self, num_examples: int) -> Tuple:
        return tuple(self.value for _ in range(num_examples))


class ProgInputGrids(Program):
    def __init__(self, grids: Tuple[Grid, ...]):
        self.grids = grids

    def __str__(self):
        return "input"

    def __repr__(self):
        return str(self)

    def evaluate(self, num_examples: int) -> Tuple:
        assert num_examples == len(self.grids)
        return self.grids
