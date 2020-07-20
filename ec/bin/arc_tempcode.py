# ------------------------------------------------------------
# VERSION THAT SYNTHESIZES AN ADD3 primitive
# singularity exec container.img python bin/arc_tempcode.py --testingTimeout 2
# ------------------------------------------------------------

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

from dreamcoder.domains.arc.arcPrimitives import tgrid, map_primitives,grid_primitives, color_primitives, ArcExample, _gridempty
from dreamcoder.domains.arc.makeArcTasks import make_tasks_anshula
from dreamcoder.domains.arc.main import ArcFeatureNN

random.seed(0)
# create primitives

def _incr(x): return x + 1

primitives = [
    # Primitive(name in Ocaml, type, name in Python)
    Primitive("incr", arrow(tint, tint), _incr)
] + color_primitives + grid_primitives

# create grammar
grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=6, 
    # activation='tanh',
    aic=.1, # LOWER THAN USUAL, to incentivize making primitives
    iterations=6, 
    # recognitionTimeout=60, 
    # featureExtractor=ArcFeatureNN,
    a=3, 
    maximumFrontier=10, 
    topK=1, 
    pseudoCounts=30.0,
    # helmholtzRatio=0.5, 
    # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
    solver='python',
    # CPUs=numberOfCPUs()
    )


# helper function that will 
#   add some number `N` to a pseudo-random number:
# The return value is a dictionary format 
#   we will use to store the inputs and the outputs for each task.
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
def add3(): return addN(3)
def add6(): return addN(6)
def add9(): return addN(9)
def add12(): return addN(12)

training_examples = [
    {"name": "add1", "examples": [add1() for _ in range(5000)]},
    {"name": "add3", "examples": [add3() for _ in range(5000)]},
    {"name": "add9", "examples": [add9() for _ in range(5000)]},
    {"name": "add6", "examples": [add6() for _ in range(5000)]}
]

training = [get_tint_task(item) for item in training_examples]

# Testing data
testing_examples = [
    {"name": "add12", "examples": [add12() for _ in range(500)]},
]
testing = [get_tint_task(item) for item in testing_examples]

# training2, testing2 = make_tasks_anshula()
# training  = training + training2
# testing = testing + testing2

for ex in training:
    print("training", ex)
# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
