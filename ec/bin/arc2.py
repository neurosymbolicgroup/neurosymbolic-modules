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

from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives, ArcExample, _gridempty
from dreamcoder.domains.arc.arcInput import load_task

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

# task that takes in grid and outputs blank grid of same shape as INPUT
task_blank_in = Task(
        task_name + "BLANK_IN",
        arrow(tgrid, tgrid),
        [((ArcExample(training_example["input"]),), _gridempty(ArcExample(training_example["input"]))) for training_example in d["train"]]
    )


array1_in = [[3, 1, 2], [3, 1, 2], [3, 1, 2]]
array1_out = [[4, 5, 2], [4, 5, 2], [4, 5, 2]]
arc1_in = ArcExample(array1_in)
arc1_out = ArcExample(array1_out)
should_be = arc1_in.map_i_to_j(3, 4).map_i_to_j(1, 5)
assert arc1_out == should_be, 'incorrect example created'

 # task that takes in grid and outputs blank grid of same shape as INPUT 
task_1 = Task(
        task_name + "FIRST_TRAINING_EXAMPLE",
        arrow(tgrid, tgrid),
        [((arc1_in,), arc1_out)])

print(task_1.examples)

training = [task_identity, task_blank_in, task_1]

testing = [task_identity]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
