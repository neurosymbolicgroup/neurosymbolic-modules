import datetime
import os
import random

# import binutil

from ec.dreamcoder.dreamcoder import commandlineArguments, ecIterator
from ec.dreamcoder.grammar import Grammar
from ec.dreamcoder.program import Primitive
from ec.dreamcoder.task import Task
from ec.dreamcoder.type import arrow, tint
from ec.dreamcoder.utilities import numberOfCPUs

from ec.dreamcoder.domains.arc.arcInput import export_tasks
from ec.dreamcoder.domains.arc.arcInput import export_dc_demo, make_consolidation_dict
from ec.dreamcoder.domains.arc.makeTasks import get_arc_task
from ec.dreamcoder.domains.arc.main import ArcNet

from ec.dreamcoder.domains.arc.arcPrimitives import *
from ec.dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from ec.dreamcoder.domains.arc.arcPrimitives import generate_ocaml_primitives

from ec.dreamcoder.domains.arc.recognition_test import run_shuffle

# set the primitives to work with
primitives = [
        # p['object'],
        p['x_mirror'],
        # p['y_mirror'],
        p['rotate_cw'],
        # p['rotate_ccw'],
        p['left_half'],
        # p['right_half'],
        # p['top_half'],
        # p['bottom_half'],
        p['overlay'],
        p['combine_grids_vertically'],
        # p['combine_grids_horizontally'], 
        p['input'],
        # p['output'],
    ]

# make a starting grammar to enumerate over
grammar = Grammar.uniform(primitives)

# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, 
    aic=0.1,
    iterations=1, 
    recognitionTimeout=120, 
    # featureExtractor=ArcNet,
    a=3, 
    maximumFrontier=10, 
    topK=5, 
    pseudoCounts=30.0,
    structurePenalty=0.1,
    solver='python'
    )

symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 345, 359, 360, 379, 371, 384]
# sometimes we have the type of the task be arrow(tinput, toutput). For this, we
# want the type to simply be arrow(tinput, tgrid). See
# dreamcoder/domains/arc/makeTasks.py
training = [get_arc_task(i, use_toutput=False) for i in symmetry_tasks]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=[],
                       outputPrefix='./ec/experimentOutputs/arc/',
                       **args)

# run the DreamCoder learning process for the set number of iterations
for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))
    solved_tasks = result.frontiersOverTime[i]
    # print(f"solved: {solved}")

# consolidation_dict = make_consolidation_dict(result)
# export_dc_demo('/home/salford/to_copy/arc_demo_9.json', training, consolidation_dict)

