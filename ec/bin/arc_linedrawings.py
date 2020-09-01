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
# from dreamcoder.domains.arc.main import ArcNet2

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.makeTasks_testing import make_tasks_getobjectcolor
from dreamcoder.domains.arc.recognition_test import *

run_test_tasks()
quit()

primitives = [
        p['input'], 
        p['0'], p['objects'], p['get'],

        p['color_in_grid'], 
        p['color'],

        p['overlay'],

        p['draw_line_slant_up'],
        p['draw_line_slant_down'], 

        # p['draw_connecting_line'],

        # p['map'], p['zip'], 


        # p['color0'], p['flood_fill'],

        # p['dir45'], p['dir315'],
        # p['draw_line'], 



        # p['overlay'],
        # p['zip'],
        # p['stack_no_crop'], p['compare'],
        # p['filter_list'],

        # p['filter_color'], 
        # p['colors'], p['area'],
        # p['sortby'],
        # p['get_object'], 

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
    enumerationTimeout=20, 
    # activation='tanh',
    aic=.1, # LOWER THAN USUAL, to incentivize making primitives
    iterations=1, 
    # recognitionTimeout=60, 
    # featureExtractor=ArcNet2,
    a=3, 
    maximumFrontier=10, 
    topK=1, 
    pseudoCounts=30.0,
    # helmholtzRatio=0.5, 
    # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
    solver='python'
    # CPUs=5
    )

training = [get_arc_task(i) for i in [36]]
# training = [get_arc_task(i) for i in range(0, 400)]

# export_tasks('/home/salford/to_copy/', training)

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
