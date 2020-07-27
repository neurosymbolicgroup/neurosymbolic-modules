# ---------------------------------------------------------
# singularity exec container.img python -u bin/arc_comprehensive.py -t 16000 -i 20 -R 4000
# ------------------------------------------------------------

import datetime
import os
import random

import numpy as np

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.grammar import Grammar

from dreamcoder.domains.arc.arcPrimitives import primitives
from dreamcoder.domains.arc.makeArcTasks import full_arc_task
from dreamcoder.domains.arc.main import ArcFeatureNN, ArcNet
from dreamcoder.domains.arc.test import test
from dreamcoder.domains.arc.robustfill import run

# create grammar
grammar = Grammar.uniform(primitives)

# generic command line options
args = commandlineArguments(
    enumerationTimeout=60,
    activation='tanh',
    aic=0.1,
    iterations=1000,
    recognitionTimeout=120,
    featureExtractor=ArcNet,
    a=3,
    maximumFrontier=10,
    topK=2,
    pseudoCounts=30.0,
    helmholtzRatio=0.5,
    structurePenalty=1.,
    solver='python',
    CPUs=numberOfCPUs()/3,
    auxiliary=True)


training = full_arc_task()
testing = []

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       outputPrefix='/experimentOutputs/logo/',
                       **args)


r = None
for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))

    if result.hitsAtEachWake[-1] == len(training):
        print('solved all tasks after {} iterations! quitting'.format(i +1))
        r = result
        # print('r: {}'.format(r))
        break


# test_recognition(result)
