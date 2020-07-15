# FROM ADD3,6,9 figure out ADDN
import datetime
import os
import random

import numpy as np

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.grammar import Grammar

from dreamcoder.domains.arc.arcPrimitives import primitives
from dreamcoder.domains.arc.makeArcTasks import get_tasks
from dreamcoder.domains.arc.main import ArcFeatureNN
from dreamcoder.domains.arc.test import test_recognition

# create grammar
grammar = Grammar.uniform(primitives)

# generic command line options
args = commandlineArguments(
    enumerationTimeout=60, activation='tanh',
    aic=0.1,
    iterations=1000, recognitionTimeout=120,
    featureExtractor=ArcFeatureNN,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    solver='python',
    CPUs=numberOfCPUs(),
    auxiliary=True)


# _, training, testing = test_ring_task()
training, testing = get_tasks()

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)


r = None
for i, result in enumerate(generator):
    print('ecIterator count {}'.format(i))

    if result.hitsAtEachWake[-1] == len(training):
        print('solved all tasks after {} iterations! quitting'.format(i +1))
        r = result
        # print('r: {}'.format(r))
        break


test_recognition(result)
