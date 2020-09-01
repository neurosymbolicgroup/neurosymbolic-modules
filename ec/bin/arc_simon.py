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

# run_test_tasks()
run_shuffle()
assert False, 'just testing'


primitives = [p['objects2'], p['T'], p['F'],
        p['construct_mapping4'], p['place_into_grid'],
        p['rotation_invariant'], 
        p['color_transform'],
        p['left_half'],
        p['right_half'], p['map_i_to_j'], p['overlay'], p['map'],
        p['combine_grids_horizontally'], p['combine_grids_vertically'],
        p['contains_color'], p['filter_list'], p['reverse'],
        p['output'], p['input'], p['area'], p['color'], 
        p['construct_mapping2'], p['size_invariant'], p['place_into_grid'],
        p['construct_mapping'], p['construct_mapping3'],
        p['construct_mapping4'],
        p['color_invariant'], p['rows'], p['columns'],
        p['vstack'], p['hstack'], p['place_into_input_grid'],
        ]

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

training = [get_arc_task(i) for i in range(0, 400)] 
# training = [get_arc_task(i) for i in [79, 126, 148, 305, 168, 329, 338, 11, 14, 15, 27, 47, 55, 80, 81, 94, 103, 127, 132, 157, 159, 166, 185, 219, 229, 265, 281, 316, 325, 330, 333, 343, 351, 367, 368, 398, 264, 72, 234, 261, 301, 102, 85]]
# training = [get_arc_task(i) for i in [11, 14, 15, 27, 55, 72, 80, 81, 94, 103, 159, 219, 229, 234, 261, 265, 281, 301, 316, 330, 343, 351]]
# training = [get_arc_task(i) for i in [47]]

# export_tasks('/home/salford/to_copy/', training)

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
