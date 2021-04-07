import datetime
import os
import random

import binutil

# from dreamcoder.domains.text.main import LearnedFeatureExtractor
from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

# create primitives

def _incr(x): return lambda x: x + 1
def _incr2(x): return lambda x: x + 2

primitives = [
    Primitive("incr", arrow(tint, tint), _incr),
    Primitive("incr2", arrow(tint, tint), _incr2),
]

# create grammar

grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, 
    activation='tanh',
    iterations=10, 
    recognitionTimeout=3600,
    a=3, 
    maximumFrontier=10, 
    topK=2, 
    # featureExtractor=LearnedFeatureExtractor,
    pseudoCounts=30.0,
    helmholtzRatio=0.5, 
    structurePenalty=1.,
    CPUs=numberOfCPUs())


# helper function that will 
# 	add some number `N` to a pseudo-random number:
# The return value is a dictionary format 
# 	we will use to store the inputs and the outputs for each task.
def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}

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
training_examples = [
    {"name": "add1", "examples": [add1() for _ in range(5000)]},
    {"name": "add2", "examples": [add2() for _ in range(5000)]},
    {"name": "add3", "examples": [add3() for _ in range(5000)]},
]
training = [get_tint_task(item) for item in training_examples]

# Testing data
def add4(): return addN(4)
testing_examples = [
    {"name": "add4", "examples": [add4() for _ in range(500)]},
]
testing = [get_tint_task(item) for item in testing_examples]


# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))













