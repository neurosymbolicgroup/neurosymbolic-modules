# FROM ADD3,6,9 figure out ADDN
import datetime
import os
import random

import numpy as np

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.utilities import numberOfCPUs

from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives
from dreamcoder.domains.arc.arcInput import load_task

# create primitives

def _incr(x): return x + 1
def _gridempty(l): return np.zeros(np.array(l).shape).astype(int).tolist()
def _gridempty(l): return np.zeros(np.array(l).shape).astype(int).tolist()


primitives =  [
    # Primitive(name in Ocaml, type, name in Python)
    # Primitive("incr", arrow(tint, tint), _incr),
    Primitive("gridempty", arrow(tlist(tint), tlist(tint)), _gridempty)

]# + primitives

# create grammar
grammar = Grammar.uniform(primitives)

# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=1, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    CPUs=numberOfCPUs())


task_name = "0d3d703e"
d = load_task(task_name)

# task_dummy = Task( # regular arc task
#         task_name + "DUMMY",
#         arrow(tlist(tint), tlist(tint)),
#         [(([1,2],), [0,0]) for training_example in d["train"]]
#     )

# task_reg = Task( # regular arc task
#         task_name + "REG",
#         arrow(tlist(tint), tlist(tint)),
#         [((training_example["input"],), training_example["output"]) for training_example in d["train"]]
#     )

task_identity = Task( # input grid is same as output grid
        task_name + "IDENTITY",
        arrow(tlist(tint), tlist(tint)),
        [((training_example["input"],), training_example["input"]) for training_example in d["train"]]
    )

print(task_identity.examples)

task_blank_in = Task( # task that takes in grid and outputs blank grid of same shape as INPUT
        task_name + "BLANK_IN",
        arrow(tlist(tint), tlist(tint)),
        [((training_example["input"],), _gridempty(training_example["input"])) for training_example in d["train"]]
    )

print(task_blank_in.examples)

# task_blank_out = Task( # task that takes in grid and outputs blank grid of same shape as OUTPUT
#         task_name + "BLANK_OUT",
#         arrow(tlist(tint), tlist(tint)),
#         [((training_example["input"],), np.zeros(np.array(training_example["output"]).shape).tolist()) for training_example in d["train"]]
#     )




training = [task_identity, task_blank_in]#, task_reg, task_identity, task_blank_in]

testing = [task_identity]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
