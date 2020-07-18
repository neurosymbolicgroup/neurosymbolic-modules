import datetime
import os
import random

import binutil

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, baseType, tint 
from dreamcoder.utilities import numberOfCPUs

tlist = baseType("tlist")

def _transform(a): 
    return lambda x1: lambda x2: lambda x3: lambda x4: lambda x5: lambda x6: lambda x7: a + [x1, x2, x3, x4, x5, x6, x7]

primitives = [
    Primitive("transform", arrow(*([tlist] + [tint]*7 + [tlist])), _transform),
    Primitive("empty_list", tlist, [])
] + [Primitive(str(i), tint, i) for i in range(0, 10)]

grammar = Grammar.uniform(primitives)

args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=10, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    solver='python',
    CPUs=numberOfCPUs())

def get_task():
    inp = []
    outp = [1,2,3,4,5,6,7]

    assert outp == _transform(inp)(1)(2)(3)(4)(5)(6)(7), "bad task made"
    return Task(
        "append",
        arrow(tlist, tlist),
        [((inp,), outp)]
    )

training = [get_task()]

generator = ecIterator(grammar,
                       training,
                       **args)

for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
