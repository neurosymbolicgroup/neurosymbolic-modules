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

from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives, ArcExample
from dreamcoder.domains.arc.arcInput import load_task

# create primitives



def _incr(x): return x + 1

def _gridempty(a): return a.empty_grid()


primitives =  [
    # Primitive(name in Ocaml, type, name in Python)
    # Primitive("incr", arrow(tint, tint), _incr),
    Primitive("gridempty", arrow(tgrid, tgrid), _gridempty)

]# + primitives

# create grammar
grammar = Grammar.uniform(primitives)

# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=2, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    CPUs=numberOfCPUs())


task_name = "d07ae81c"
d = load_task(task_name)

task_identity = Task( # input grid is same as output grid
        task_name + "IDENTITY",
        arrow(tgrid, tgrid),
        [((ArcExample(training_example["input"]),), ArcExample(training_example["input"])) for training_example in d["train"]]
    )

task_identity2 = Task( # input grid is same as output grid
        task_name + "IDENTITY2",
        arrow(tgrid, tgrid),
        [((training_example["input"],), training_example["input"]) for training_example in d["train"]]
    )


# print(task_identity.examples)

task_blank_in = Task( # task that takes in grid and outputs blank grid of same shape as INPUT
        task_name + "BLANK_IN",
        arrow(tgrid, tgrid),
        [((ArcExample(training_example["input"]),), _gridempty(ArcExample(training_example["input"]))) for training_example in d["train"]]
    )


training = [task_identity, task_blank_in]

testing = [task_identity]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
