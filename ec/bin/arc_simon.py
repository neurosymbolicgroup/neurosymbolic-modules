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
from dreamcoder.domains.arc.arcInput import export_dc_demo, make_consolidation_dict
from dreamcoder.domains.arc.makeTasks import get_arc_task
# from dreamcoder.domains.arc.symmetry import run as run_test_tasks
from dreamcoder.domains.arc.tasks_8_26 import run as run_test_tasks
from dreamcoder.domains.arc.main import ArcNet

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.arcPrimitives import generate_ocaml_primitives

# from dreamcoder.domains.arc.makeTasks_testing import make_tasks_getobjectcolor
from dreamcoder.domains.arc.recognition_test import run_shuffle

# run_test_tasks()
# generate_ocaml_primitives()
# run_shuffle()

primitives = [
        p['object'],
        p['x_mirror'], p['y_mirror'],
        p['rotate_ccw'], p['rotate_cw'],
        p['left_half'], p['right_half'], 
        p['top_half'], p['bottom_half'],
        p['overlay'],
        p['combine_grids_horizontally'], p['combine_grids_vertically'],
        p['input'],
    ]

grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, 
    # activation='tanh',
    aic=.1, # LOWER THAN USUAL, to incentivize making primitives
    iterations=1, 
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

# primitives = primitives0
# training = [get_arc_task(56)]
training = [get_arc_task(i) for i in range(0, 400)] 
# copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
# copy_two_tasks = [103, 166, 55, 166, 103, 47, 185, 398, 102] + [86]
symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 346, 359, 360, 379, 371, 384]
training = [get_arc_task(i) for i in symmetry_tasks]
# training = [get_arc_task(i) for i in [30, 52]]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)
for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))
    for task in training:
        task.arc_iteration += 1

consolidation_dict = make_consolidation_dict(result)
export_dc_demo('/home/salford/to_copy/arc_demo_8.json', training, consolidation_dict)

