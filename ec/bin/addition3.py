# FROM ADD3,6,9 figure out ADDN
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

def _rpt(n, f, x0): 
	"""
	Apply function f to x0 repeatedly
	Specifically, n times
	"""
	r = lambda n, f, x0: x0 if n==0 else f( r(n-1, f, x0) )
	return r



primitives = [
    # Primitive(name in Ocaml, type, name in Python)
    Primitive("incr", arrow(tint, tint), _incr),
    Primitive("rpt", arrow(tint, arrow(tint, tint), tint, tint), _rpt),
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
def addN(n=None):
    x = random.choice(range(500))
    if n==None: 
        n = random.choice(range(500))
        return {"i": (x,n,), "o": x + n}
    else:
        return {"i": (x,), "o": x + n}

# Each task will consist of 3 things:
# 1. a name
# 2. a mapping from input to output type (e.g. `arrow(tint, tint)`)
# 3. a list of input-output pairs
def get_tint_task_1arg(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [(ex["i"], ex["o"]) for ex in item["examples"]],
    )

def get_tint_task_2arg(item):
    return Task(
        item["name"],
        arrow(tint, arrow(tint, tint)),
        [(ex["i"], ex["o"]) for ex in item["examples"]],
    )

# Training data
training_examples = [
    {"name": "add3", "examples": [addN(3) for _ in range(3)]},
    {"name": "add9", "examples": [addN(9) for _ in range(3)]},
    {"name": "add6", "examples": [addN(6) for _ in range(3)]}
]

training = [get_tint_task_1arg(item) for item in training_examples]
print('training: {}'.format(training))

# Testing data
testing_examples = [
    # {"name": "add12", "examples": [addN(12) for _ in range(500)]}
    {"name": "addN", "examples": [addN() for _ in range(5)]},
]

testing = [get_tint_task_2arg(item) for item in testing_examples]
print(training[0].name, training[0].examples[0:3])
print(training[1].name, training[1].examples[0:3])
print(training[2].name, training[2].examples[0:3])

print(testing[0].name, testing[0].examples[0:3])

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
