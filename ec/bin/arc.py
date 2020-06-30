# FROM ADD3,6,9 figure out ADDN
import datetime
import os
import random

import numpy as np

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.utilities import numberOfCPUs

from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives
from dreamcoder.domains.arc.arcInput import load_task

# create primitives

def _incr(x): return x + 1

def _gridempty(l): return np.zeros(np.array(l).shape).astype(int).tolist()

def _map3to4(l): 
    m = np.copy(l)
    m[m==3]=4
    return m.tolist()
def _map1to5(l): 
    m = np.copy(l)
    m[m==1]=5
    return m.tolist()
def _map2to6(l): 
    m = np.copy(l)
    m[m==2]=6
    return m.tolist()

def _mapitoj_python(i,j,l):
    m = np.copy(l)
    m[m==i]=j
    return m.tolist()

def _mapitoj(i): 
    return lambda j: lambda l: _mapitoj_python(i,j,l)


primitives =  [
    # Primitive(name in Ocaml, type, name in Python)

    Primitive("gridempty", arrow(tlist(tint), tlist(tint)), _gridempty),

    # Primitive("map3to4", arrow(tlist(tint), tlist(tint)), _map3to4),
    # Primitive("map1to5", arrow(tlist(tint), tlist(tint)), _map1to5),
    Primitive("map2to6", arrow(tlist(tint), tlist(tint)), _map2to6),

    Primitive("mapitoj", arrow(tint, tint, tlist(tint), tlist(tint)), _mapitoj),

    # Primitive("0", tint, 0),
    Primitive("1", tint, 1),
    Primitive("2", tint, 2),
    Primitive("3", tint, 3),
    Primitive("4", tint, 4),
    Primitive("5", tint, 5),
    Primitive("6", tint, 6),
    # Primitive("7", tint, 7),
    # Primitive("8", tint, 8),
    # Primitive("9", tint, 9)

]# + primitives

# create grammar
grammar = Grammar.uniform(primitives)

# generic command line options
args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=1, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    CPUs=numberOfCPUs())


task_name = "0d3d703e"
d = load_task(task_name)

# task_dummy = Task( # regular arc task
#         task_name + "DUMMY",
#         arrow(tlist(tint), tlist(tint)),
#         [(([1,2],), [0,0]) for training_example in d["train"]]
#     )

# task_reg = Task( # regular arc task
#         task_name + "REG",
#         arrow(tlist(tint), tlist(tint)),
#         [((training_example["input"],), training_example["output"]) for training_example in d["train"]]
#     )

task_identity = Task( # input grid is same as output grid
        task_name + "IDENTITY",
        arrow(tlist(tint), tlist(tint)),
        [((training_example["input"],), training_example["input"]) for training_example in d["train"]]
    )

print(task_identity.examples)

task_blank_in = Task( # task that takes in grid and outputs blank grid of same shape as INPUT
        task_name + "BLANK_IN",
        arrow(tlist(tint), tlist(tint)),
        [((training_example["input"],), _gridempty(training_example["input"])) for training_example in d["train"]]
    )

print(task_blank_in.examples)

task_1 = Task( # task that takes in grid and outputs blank grid of same shape as INPUT
        task_name + "FIRST_TRAINING_EXAMPLE",
        arrow(tlist(tint), tlist(tint)),
        [(([[3, 1, 2], [3, 1, 2], [3, 1, 2]],), [[4, 5, 6], [4, 5, 6], [4, 5, 6]])]
        # [(([[3, 1, 2], [3, 1, 2], [3, 1, 2]],), [[4, 1, 2], [4, 1, 2], [4, 1, 2]])]
        # [((training_example["input"],), training_example["output"]) for training_example in [d["train"][0]]]
    )

print(task_1.examples)


# task_2 = Task( # task that takes in grid and outputs blank grid of same shape as INPUT
#         task_name + "COLOR_PIXEL",
#         arrow(tint, tlist(tint), tlist(tint)),
#          [(([1,2,3,4,5],), [[4]])]
#         # [(([[3, 1, 2], [3, 1, 2], [3, 1, 2]],), [[4, 1, 2], [4, 1, 2], [4, 1, 2]])]
#         # [((training_example["input"],), training_example["output"]) for training_example in [d["train"][0]]]
#     )

# print(task_1.examples)


training = [task_identity, task_blank_in, task_1]#, task_reg, task_identity, task_blank_in]

testing = [task_identity]

# iterate over wake and sleep cycles for our task
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))