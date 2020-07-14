# FROM ADD3,6,9 figure out ADDN
import datetime
import os
import random

import numpy as np

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.grammar import Grammar

from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives, ArcExample, _gridempty
from dreamcoder.domains.arc.makeArcTasks import make_tasks, make_tasks_anshula
from dreamcoder.domains.arc.main import ArcFeatureNN

# create grammar
grammar = Grammar.uniform(primitives)

# simon's command line options
# args = commandlineArguments(
#     enumerationTimeout=10, activation='tanh',
#     aic=0.1,
#     iterations=2, recognitionTimeout=60,
#     # featureExtractor=ArcFeatureNN,
#     a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
#     helmholtzRatio=0.5, structurePenalty=1.,
#     solver='python',
#     CPUs=numberOfCPUs())

# anshula's command line options
args = commandlineArguments(
    enumerationTimeout=10, 
    activation='tanh',
    aic=1.,
    iterations=2, 
    recognitionTimeout=60, 
    # featureExtractor=ArcFeatureNN,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=.001,
    solver='python',
    # CPUs=numberOfCPUs()
    )

# simon's tasks
# training, testing = make_tasks()

# anshula's tasks
training, testing = make_tasks_anshula()

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)

for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
