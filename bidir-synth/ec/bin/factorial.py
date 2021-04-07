import datetime
import os
import random

import binutil

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

# create primitives

def _incr(x): return lambda x: x + 1
def _incr2(x): return lambda x: x + 2
# def _add(x,y): return lambda x: x + y

# def _mult1(x): return lambda x: x * 1
# def _mult2(x): return lambda x: x * 2
# def _mult3(x): return lambda x: x * 3

primitives = [
    # Primitive(name in Ocaml, type, name in Python)
    Primitive("incr", arrow(tint, tint), _incr),
    Primitive("incr2", arrow(tint, tint), _incr2),
    # Primitive("mult1", arrow(tint, tint), _mult1),
    # Primitive("mult2", arrow(tint, tint), _mult2),
    # Primitive("mult3", arrow(tint, tint), _mult3),

    # Primitive("+", arrow(tint, tint), _add),
]

# create grammar

grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=10, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    CPUs=numberOfCPUs())


# helper function that will 
#   add some number `N` to a pseudo-random number:
# The return value is a dictionary format 
#   we will use to store the inputs and the outputs for each task.
def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}

def multN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x * n}

# Each task will consist of 3 things:
# 1. a name
# 2. a mapping from input to output type (e.g. `arrow(tint, tint)`)
# 3. a list of input-output pairs
def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )

# Training data
def add1(): return addN(1)
def add2(): return addN(2)
def add3(): return addN(3)
def add6(): return addN(6)

def mult1(): return multN(1)
def mult2(): return multN(2)
def mult3(): return multN(3)
def mult4(): return multN(4)

# training_examples = [
#   # 5000 examples of multiplying a number by 1
#     {"name": "mult1", "examples": [mult1() for _ in range(5000)]},
#   # 5000 examples of multiplying a number by 2
#     {"name": "mult2", "examples": [mult2() for _ in range(5000)]},
#   # 5000 examples of multiplying a number by 3
#     {"name": "mult3", "examples": [mult3() for _ in range(5000)]},
# ]

training_examples = [
    {"name": "add2", "examples": [add2() for _ in range(5000)]},
    {"name": "add3", "examples": [add3() for _ in range(5000)]}
]

training = [get_tint_task(item) for item in training_examples]

# Testing data
testing_examples = [
    {"name": "add6", "examples": [add6() for _ in range(500)]},
]
testing = [get_tint_task(item) for item in testing_examples]


# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
