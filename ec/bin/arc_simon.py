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
from dreamcoder.domains.arc.makeTasks import get_arc_task
# from dreamcoder.domains.arc.symmetry import run as run_test_tasks
from dreamcoder.domains.arc.tasks_8_26 import run as run_test_tasks
from dreamcoder.domains.arc.main import ArcNet

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.makeTasks_testing import make_tasks_getobjectcolor
from dreamcoder.domains.arc.recognition_test import *

run_test_tasks()
assert False, 'just testing'

primitives = [p['objects2'], p['True'], p['False'],
        p['rotation_invariant'], p['construct_mapping'], p['size_invariant'],
        p['no_invariant'], p['color_invariant'], p['rows'], p['columns']]

grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=60, 
    # activation='tanh',
    aic=.1, # LOWER THAN USUAL, to incentivize making primitives
    iterations=2, 
    recognitionTimeout=120, 
    featureExtractor=ArcNet,
    a=3, 
    maximumFrontier=10, 
    topK=1, 
    pseudoCounts=30.0,
    # helmholtzRatio=0.5, 
    # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
    solver='python'
    # CPUs=5
    )

# training = [get_arc_task(i) for i in range(0, 400)] testing a really 
training = [get_arc_task(i) for i in [79, 126, 148, 305, 168, 329, 338, 11, 14, 15, 27, 47, 55, 80, 81, 94, 103, 127, 132, 157, 159, 166, 185, 219, 229, 265, 281, 316, 325, 330, 333, 343, 351, 367, 368, 398, 264, 72, 234, 261, 102, 301, 102, 85]]

# export_tasks('/home/salford/to_copy/', training)

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
