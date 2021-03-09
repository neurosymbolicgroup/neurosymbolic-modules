import datetime
import os
import random

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive, Index, Abstraction, Application
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

from dreamcoder.domains.arc.arcInput import export_tasks
from dreamcoder.domains.arc.arcInput import export_dc_demo, make_consolidation_dict
from dreamcoder.domains.arc.makeTasks import get_arc_task
from dreamcoder.domains.arc.main import ArcNet

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.arcPrimitives import generate_ocaml_primitives

from dreamcoder.domains.arc.recognition_test import run_shuffle

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
                       outputPrefix='./experimentOutputs/arc/',
                       **args)

# run the DreamCoder learning process for the set number of iterations
for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))
    fot = result.frontiersOverTime
    task384 = get_arc_task(384, use_toutput=False)
    print(f"task384: {task384}")
    frontiers384 = fot[task384]
    print(f"frontiers384: {frontiers384}")
    first_frontier = frontiers384[0]
    first_entry = first_frontier.entries[0]
    program = first_entry.program
    print()
    print()
    print(f"program: {program}")
    print(type(program))
    abstraction = program
    body = abstraction.body
    print(f"body: {body}")
    print(f"type(body): {type(body)}")
    f1 = body.f
    print(f"f1: {f1}")
    print(f"type(f1): {type(f1)}")
    x1 = body.x
    print(f"type(x1): {type(x1)}")
    f2 = f1.f
    print(f"f2: {f2}")
    print(f"type(f2): {type(f2)}")
    x2 = f1.x
    print(f"x2: {x2}")
    print(f"type(x2): {type(x2)}")
    f3 = x2.f
    x3 = x2.x
    print(f"x3: {x3}")
    print(f"type(x3): {type(x3)}")
    print(f"x3.i: {x3.i}")

    f = Application(p['input'], Index(0))
    f = Application(p['rotate_cw'], f)
    f2 = Application(p['rotate_cw'], f)
    program = Abstraction(f2)
    print(f"program: {program}")
    task86 = get_arc_task(86, use_toutput=False)
    solved = task86.check(program, timeout=10000)
    print(f"solved: {solved}")


    assert False


# consolidation_dict = make_consolidation_dict(result)
# export_dc_demo('/home/salford/to_copy/arc_demo_9.json', training, consolidation_dict)

