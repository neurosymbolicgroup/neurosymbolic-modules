# FROM ADD X, FIGURE OUT ADD X Y
import datetime
import os
import random

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

# create primitives

def _incr(x): return lambda x: x + 1

primitives = [
    # Primitive(name in Ocaml, type, name in Python)
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


# helper function
def addXY():
    x = random.choice(range(500))
    y = random.choice(range(500))
    return {"i": (x,y), "o": x + y}

# Each task will consist of 3 things:
# 1. a name
# 2. a mapping from input to output type (e.g. `arrow(tint, tint)`)
# 3. a list of input-output pairs
def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, arrow(tint, tint)),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


training_examples = [
    {"name": "addXY", "examples": [addXY() for _ in range(5000)]}
]

training = [get_tint_task(item) for item in training_examples]

# Testing data
testing_examples = [
    {"name": "addXY", "examples": [addXY() for _ in range(500)]},
]
testing = [get_tint_task(item) for item in testing_examples]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
