from bidir.primitives.functions import Function
from bidir.primitives.types import Grid
from typing import Any, List, Tuple


class Program:
    def evaluate(self, inputs: Tuple) -> Any:
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

    def evaluate(self, inputs: Tuple) -> Any:
        arg_vals = [arg.evaluate(inputs) for arg in self.args]
        return self.fn.fn(*arg_vals)


class ProgConstant(Program):
    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"CONST: {self}"

    def evaluate(self, inputs: Any) -> Any:
        return self.value


class ProgInput(Program):
    def __init__(self, ix: int):
        self.ix = ix

    def __str__(self):
        return f"${self.ix}"

    def __repr__(self):
        return str(self)

    def evaluate(self, inputs) -> Grid:
        return inputs[self.ix]
