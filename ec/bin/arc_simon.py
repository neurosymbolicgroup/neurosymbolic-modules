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
from dreamcoder.domains.arc.symmetry import run as run_test_tasks
from dreamcoder.domains.arc.main import ArcNet

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.makeTasks_testing import make_tasks_getobjectcolor
from dreamcoder.domains.arc.recognition_test import *

run_test_tasks()
# assert False, 'just testing'

primitives = [p['input'], p['object'], p['overlay'], p['objects'],
        p['objects_by_color'], p['filter_list'], p['get'], p['0'],
        p['has_y_symmetry'], p['has_x_symmetry'], p['has_rotational_symmetry'],
        p['rotate_ccw'], 
        p['combine_grids_vertically'], p['combine_grids_horizontally'],
        p['x_mirror'], p['y_mirror'], 
        p['top_half'], p['bottom_half'], p['left_half'], p['right_half']]


# create grammar
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
    solver='python',
    CPUs=5
    )

# training = [get_arc_task(i) for i in [379, 139, 86, 149, 154, 209, 171, 163, 38,
    # 112, 115, 173]]
training = [get_arc_task(i) for i in range(0, 400)]

# export_tasks('/home/salford/to_copy/', training)

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
