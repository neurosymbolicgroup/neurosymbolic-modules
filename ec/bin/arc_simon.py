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

from dreamcoder.domains.arc.arcInput import export_tasks
from dreamcoder.domains.arc.makeTasks import make_arc_task
from dreamcoder.domains.arc.makeTasks import run as run_test_tasks
from dreamcoder.domains.arc.main import ArcNet2

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.makeTasks_testing import make_tasks_getobjectcolor
from dreamcoder.domains.arc.recognition_test import *

print('hi')
run_test_tasks()
assert False, 'Simon testing, feel free to remove'


# for task 1
# primitives = [p['map_i_to_j'], p['input']]
# colors = [p['color' + str(i)] for i in range(10)]
# primitives = primitives + colors

# for task 2
primitives = ['get', 'objects', 'input', 'absolute_grid', 'pixels']
primitives = [p[i] for i in primitives]
ints = [p[str(i)] for i in range(5)]
primitives = primitives + ints

# combine tasks, hopefully won't solve
primitives = ['get', 'objects', 'input', 'absolute_grid', 'pixels',
'map_i_to_j']
primitives = [p[i] for i in primitives]
ints = [p[str(i)] for i in range(5)]
colors = [p['color' + str(i)] for i in range(10)]
primitives = primitives + ints + colors

# create grammar
grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=7, 
    # activation='tanh',
    aic=.1, # LOWER THAN USUAL, to incentivize making primitives
    iterations=2, 
    # recognitionTimeout=60, 
    featureExtractor=ArcNet2,
    a=3, 
    maximumFrontier=10, 
    topK=1, 
    pseudoCounts=30.0,
    # helmholtzRatio=0.5, 
    # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
    solver='python',
    CPUs=5
    )

# training = [task1(i) for i in range(10)]
# training = [task2(i) for i in range(10)]
training = [task1(i) for i in range(10)] + [task2(i) for i in range(10)]

export_tasks('/home/salford/to_copy/', training)

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
