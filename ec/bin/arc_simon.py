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

run_test_tasks()
# generate_ocaml_primitives()
# run_shuffle()
# assert False

primitives = [
        # p['objects2'],
        # p['T'], p['F'],
        p['input'],
        p['object'],
        # p['rotation_invariant'],
        # p['size_invariant'],
        # p['color_invariant'],
        # p['no_invariant'],
        # p['place_into_input_grid'],
        # p['place_into_grid'],
        # p['rows'],
        # p['columns'],
        p['output'],
        # p['size'],
        # p['area'],
        # p['construct_mapping'],
        # p['vstack'],
        # p['hstack'],
        # p['construct_mapping2'],
        # p['construct_mapping3'],
        p['area'],
        # p['has_y_symmetry'],
        # p['length'],
        # p['filter_list'],
        # p['contains_color'],
        # p['color2'],
        p['kronecker'],
        p['inflate'],
        p['deflate'],
        p['2'],
        p['3'],
        p['num_colors'],
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

# training = [get_arc_task(i) for i in range(0, 400)] 
# copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
copy_two_tasks = [103, 55, 166, 47, 185, 398, 102]
# symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 346, 359, 360, 379, 371, 384]
inflate_tasks = [0, 194, 216, 222, 268, 288, 306, 383]
training = [get_arc_task(i) for i in inflate_tasks]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./experimentOutputs/arc/',
                       **args)

for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))


