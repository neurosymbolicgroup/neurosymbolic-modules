import datetime
import os
import random

import binutil  # needed for importing things properly

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

from dreamcoder.domains.arc.arcInput import export_tasks
from dreamcoder.domains.arc.arcInput import export_dc_demo, make_consolidation_dict
from dreamcoder.domains.arc.makeTasks import get_arc_task, get_eval_tasks
from dreamcoder.domains.arc.task_testing import check_tasks
from dreamcoder.domains.arc.main import ArcNet, check_test_accuracy

from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.arcPrimitives import generate_ocaml_primitives
from dreamcoder.domains.arc.test import test
from dreamcoder.domains.arc.arcnet_test import generate_dataset, train


def arc_compare():
    primitives = [
        p['rotate_cw'],
        p['rotate_ccw'],
        p['hflip'],
        p['vflip'],
        p['hstack_pair']
        p['vstack_pair'],
        p['top_half'],
    ]

    # twenty_four_primitives = [
    #     p['add'],
    #     p['sub'],
    #     p['mul'],
    #     p['div'],
    # ]

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
        solver='python')

    task_nums = [
        38, 82, 86, 105, 115, 139, 141, 149, 151, 154, 163, 171, 178, 209,
        210, 240, 248, 310, 379, 178,
    ]

    training = [get_arc_task(i) for i in task_nums]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    # run the DreamCoder learning process for the set number of iterations
    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))


arc_compare()
