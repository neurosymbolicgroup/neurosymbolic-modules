import random
import datetime
import os

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.utilities import numberOfCPUs

# create primitives


def _incrList(x): return lambda x: [x[0]]

def _incr(x): return lambda x: x + 1


primitives = [
    # Primitive(name in Ocaml, type, name in Python)
Primitive("incrList", arrow(tlist(tint), tlist(tint)), _incrList),
Primitive("incr", arrow(tint, tint), _incr)
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

def listFunction():
    x = random.choice(range(500))
    return {"i": [x, 1, 2], "o": [x]}


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

def get_tlist_task(item):
    return Task(
        item["name"],
        arrow(tlist(tint), tlist(tint)),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


# Training data
def add1(): return addN(1)
def add3(): return addN(3)
def add6(): return addN(6)


training_examples = [
    {"name": "add3", "examples": [add3() for _ in range(5000)]}
]

list_training_examples = [
        {"name": "addList", "examples": [listFunction() for _ in range(500)]}
]

training = [get_tint_task(item) for item in training_examples]
training += [get_tlist_task(item) for item in list_training_examples]

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
