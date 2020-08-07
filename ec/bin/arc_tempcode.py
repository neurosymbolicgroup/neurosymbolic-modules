# ------------------------------------------------------------
# VERSION THAT SYNTHESIZES AN ADD3 primitive
# singularity exec container.img python bin/arc_tempcode.py --testingTimeout 2
# ------------------------------------------------------------

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

from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.makeTasks_testing import make_tasks_getobjectcolor

random.seed(0)
# create primitives


primitives = [get_prim, color_prim, objects_prim] + ints + colors
# primitives = [get_prim, objects_prim] + ints #+ colors

# create grammar
grammar = Grammar.uniform(primitives)


# generic command line options
args = commandlineArguments(
    enumerationTimeout=7, 
    # activation='tanh',
    aic=.1, # LOWER THAN USUAL, to incentivize making primitives
    iterations=2, 
    # recognitionTimeout=60, 
    # featureExtractor=ArcFeatureNN,
    a=3, 
    maximumFrontier=10, 
    topK=1, 
    pseudoCounts=30.0,
    # helmholtzRatio=0.5, 
    # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
    solver='python',
    # CPUs=numberOfCPUs()
    )


training, testing = make_tasks_getobjectcolor()
# training  = training + training2


for ex in training:
    print("training", ex)
# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
