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
from dreamcoder.domains.arc.linedrawings import run as run_test_tasks

# from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.makeTasks_testing import make_rotation_tasks
# from dreamcoder.domains.arc.recognition_test import *

# run_test_tasks()
# quit()

# print(primitive_dict)

primitives = [
        p['0'], p['objects'], p['get'],
        p['move_down'], p['draw_line_down'], p['reflect_down'],
        p['rotate_ccw'], p['rotate_cw'],

        # p['input'], 
        # p['0'], p['objects'], p['get'],
        # p['map'],

        # p['color_in_grid'], 
        # p['color'],
        # p['group_objects_by_color'],

        # p['overlay'],
        # p['stack_no_crop'],

        # p['draw_line_slant_up'],
        # p['draw_line_slant_down'], 
        # p['draw_connecting_line'], 

        # p['objects_by_color'], 
        # p['has_y_symmetry'], p['has_x_symmetry'], p['has_rotational_symmetry'],
        # p['rotate_ccw'], 
        # p['combine_grids_vertically'], p['combine_grids_horizontally'],
        # p['x_mirror'], p['y_mirror'], 
        # p['top_half'], p['bottom_half'], p['left_half'], p['right_half']
        ]


# create grammar
grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=60, 
    # activation='tanh',
    aic=0.0, # LOWER THAN USUAL, to incentivize making primitives
    iterations=2, 
    # recognitionTimeout=60, 
    # featureExtractor=ArcNet2,
    a=2, 
    maximumFrontier=1, 
    topK=1, 
    pseudoCounts=30.0,
    structurePenalty=-0.6, # HIGHER THAN USUAL, to incentivize making primitives
    solver='python'
    # helmholtzRatio=0.5, 
    # CPUs=5
    )

# training = [get_arc_task(i) for i in [36,140]]
# training = [get_arc_task(i) for i in range(0, 400)]
training = make_rotation_tasks()

# export_tasks('/home/salford/to_copy/', training)

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
