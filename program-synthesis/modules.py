from enum import Enum
import sys

class Function:
    def __init__(self, modules):
        self.modules = modules
    def execute(self, inp):
        vs = inp
        for m in self.modules:
            vs = m.execute(vs)
        return vs
    
class If(Function):
    def __init__(self, condition, function, else_function = None):
        self.condition = condition
        self.function = function
        self.else_function = else_function
    def execute(self, inp):
        if self.condition.execute(inp):
            return self.function.execute(inp)
        if self.else_function is not None:
            return self.else_function.execute(inp)
        return inp
    
class For(Function):
    def __init__(self, iterator, function):
        self.iterator = iterator
        self.function = function
    def execute(self, inp):
        vs = inp
        for i in self.iterator:
            vs = self.function.execute([vs, i])
        return vs
    
class While(Function):
    def __init__(self, condition, function):
        self.condition = condition
        self.function = function
    def execute(self, inp):
        vs = inp
        while(self.condition.execute(vs)):
            vs = self.function.execute(vs)
        return vs
    
class SetVar(Function):
    def __init__(self, name, function):
        self.name = name
        self.function = function
    def execute(self, inp):
        inp.set_var(self.name, self.function.execute(inp))
        return inp

class GetVar(Function):
    def __init__(self, name):
        self.name = name
    def execute(self, inp):
        return inp.get_var(self.name)

class Comparison(Enum):
    EQUALS = 1
    NOTEQUALS = 2
    GREATERTHAN = 3
    LESSTHAN = 4
    GREATERTHANEQ = 5
    LESSTHANEQ = 6
    
class BoolUnit(Function):
    def __init__(self, exp_1, exp_2, comp):
        self.exp_1 = exp_1
        self.exp_2 = exp_2
        self.comp = comp
    def execute(self, inp):
        if self.comp == Comparison.EQUALS:
            return self.exp_1.execute(inp) == self.exp_2.execute(inp)
        if self.comp == Comparison.NOTEQUALS:
            return self.exp_1.execute(inp) != self.exp_2.execute(inp)
        if self.comp == Comparison.GREATERTHAN:
            return self.exp_1.execute(inp) > self.exp_2.execute(inp)
        if self.comp == Comparison.LESSTHAN:
            return self.exp_1.execute(inp) < self.exp_2.execute(inp)
        if self.comp == Comparison.GREATERTHANEQ:
            return self.exp_1.execute(inp) >= self.exp_2.execute(inp)
        return self.exp_1.execute(inp) <= self.exp_2.execute(inp)

class Operation(Enum):
    PLUS = 1
    MINUS = 2
    TIMES = 3
    DIVIDEDBY = 4
    MOD = 5

class ArithUnit(Function):
    def __init__(self, exp_1, exp_2, op):
        self.exp_1 = exp_1
        self.exp_2 = exp_2
        self.op = op
    def execute(self, inp):
        if self.op == Operation.PLUS:
            return self.exp_1.execute(inp) + self.exp_2.execute(inp)
        if self.op == Operation.MINUS:
            return self.exp_1.execute(inp) - self.exp_2.execute(inp)
        if self.op == Operation.TIMES:
            return self.exp_1.execute(inp) * self.exp_2.execute(inp)
        if self.op == Operation.DIVIDEDBY:
            return self.exp_1.execute(inp) / self.exp_2.execute(inp)
        return self.exp_1.execute(inp) % self.exp_2.execute(inp)

class ListOp(Enum):
    GETIND = 1
    SETIND = 2
    APPEND = 3
    LEN = 4
    
class ListUnit(Function):
    def __init__(self, list_name, list_op, idx=-1):
        self.list_name = list_name
        self.list_op = list_op
        self.idx = idx
    
    def execute(self, inp):
        list_var = inp.get_var(self.list_name)
        if self.list_op == ListOp.GETIND:
            return list_var[self.idx.execute(inp)]
        if self.list_op == ListOp.SETIND:
            list_var[self.idx.execute(inp)] = inp
            return list_var
        if self.list_op == ListOp.APPEND:
            list_var.append(inp)
            return list_var
        return len(list_var)
    
class Value(Function):
    def __init__(self, val):
        self.val = val
    def execute(self, inp):
        return self.val
    
class Variables:
    def __init__(self, vs):
        self.vs = vs
    def set_var(self, name, value):
        self.vs[name] = value
    def get_var(self, name):
        return self.vs[name]

class ExtraOps(Function):
    def __init__(self, exp, op):
        self.exp = exp
        self.op = op
    def execute(self, inp):
        if self.op == 'to_int':
            return int(self.exp.execute(inp))
        return -1
    
# TODO: this "cheats" by just terminating the functions, will refactor code so it's less hacky
class Return(Function):
    def __init__(self, return_val):
        self.return_val = return_val
    def execute(self, inp):
        print('module-based solution:', self.return_val.execute(inp))
        sys.exit()
