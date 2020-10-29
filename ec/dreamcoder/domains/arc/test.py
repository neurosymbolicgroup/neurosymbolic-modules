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
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.makeTasks import get_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as pd
from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.domains.arc.main import ArcNet, TestRecNet, TestRecNet2


def test_construct_mapping():
    primitives = [
            pd['objects2'],
            pd['T'], pd['F'],
            pd['input'],
            pd['rotation_invariant'],
            pd['size_invariant'],
            pd['color_invariant'],
            pd['no_invariant'],
            # pd['place_into_input_grid'],
            pd['place_into_grid'],
            pd['rows'],
            pd['columns'],
            # pd['output'],
            # pd['area'],
            pd['construct_mapping'],
            pd['vstack'],
            pd['hstack'],
            # pd['construct_mapping2'],
            # pd['construct_mapping3'],
    ]

    grammar = Grammar.uniform(primitives)


    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=60, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        # recognitionTimeout=120, 
        featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        # CPUs=5
        no_consolidation=True,
        )

    copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
    training = [get_arc_task(i) for i in copy_one_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    print('should have solved all {} tasks'.format(len(copy_one_tasks)))


def test_construct_mapping2():
    primitives = [
        pd['objects2'],
        pd['T'], pd['F'],
        pd['input'],
        pd['rotation_invariant'],
        pd['size_invariant'],
        pd['color_invariant'],
        pd['no_invariant'],
        pd['place_into_input_grid'],
        pd['place_into_grid'],
        # pd['rows'],
        # pd['columns'],
        # pd['output'],
        # pd['size'],
        # pd['area'],
        pd['construct_mapping'],
        # pd['vstack'],
        # pd['hstack'],
        # pd['construct_mapping2'],
        pd['construct_mapping3'],
        pd['list_of_one'],
        pd['area'],
        pd['has_y_symmetry'],
        pd['list_length'],
        pd['filter_list'],
        pd['contains_color'],
        pd['color2'],
    ]

    grammar = Grammar.uniform(primitives)


    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=60, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        recognitionTimeout=120, 
        # featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        # CPUs=5
        no_consolidation=True,
        )

    copy_two_tasks = [103, 55, 166, 47, 185, 398, 102, 297, 352, 372]
    training = [get_arc_task(i) for i in copy_two_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    print('should have solved all {} tasks'.format(len(copy_two_tasks)))

def test_inflate():
    primitives = [
            # pd['objects2'],
            # pd['T'], pd['F'],
            pd['input'],
            pd['object'],
            # pd['rotation_invariant'],
            # pd['size_invariant'],
            # pd['color_invariant'],
            # pd['no_invariant'],
            # pd['place_into_input_grid'],
            # pd['place_into_grid'],
            # pd['rows'],
            # pd['columns'],
            pd['output'],
            # pd['size'],
            # pd['area'],
            # pd['construct_mapping'],
            # pd['vstack'],
            # pd['hstack'],
            # pd['construct_mapping2'],
            # pd['construct_mapping3'],
            pd['area'],
            pd['kronecker'],
            pd['inflate'],
            pd['deflate'],
            pd['2'],
            pd['3'],
            pd['num_colors'],
    ]

    grammar = Grammar.uniform(primitives)

    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=300, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        # recognitionTimeout=120, 
        featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        # CPUs=5
        no_consolidation=True,
        )

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

    print('should have solved all {} tasks'.format(len(inflate_tasks)))


def test_symmetry():
    primitives = [
        pd['object'],
        pd['x_mirror'],
        pd['y_mirror'],
        pd['rotate_cw'],
        pd['rotate_ccw'],
        pd['left_half'],
        pd['right_half'], 
        pd['top_half'],
        pd['bottom_half'],
        pd['overlay'],
        pd['combine_grids_vertically'],
        pd['combine_grids_horizontally'], 
        pd['input'],
        pd['output'],
    ]

    grammar = Grammar.uniform(primitives)

    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=300, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        # recognitionTimeout=120, 
        featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        # CPUs=5
        no_consolidation=True,
        )

    # symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 346, 359, 360, 379, 371, 384]
    symmetry_tasks = [30, 38, 86, 112, 115, 139, 149, 154, 163, 171, 176, 178, 209, 240, 248, 310, 359, 379, 384]

    training = [get_arc_task(i) for i in symmetry_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    print('should have solved all {} tasks'.format(len(symmetry_tasks)))

def test_rec1(ratio=0.0):
    resume = './experimentOutputs/arc/_aic=0.1_arity=3_aux=True_BO=False_CO=True_ES=1_ET=20_HR=0.0_it=1_MF=10_noConsolidation=True_pc=30.0_RT=500_RR=False_RW=False_solver=python_STM=True_L=0.001_TRR=default_K=1_topkNotMAP=False.pickle'
    symmetry_tasks = [30, 38, 86, 112, 115, 139, 149, 154, 163, 171, 176, 178, 209, 240, 248, 310, 359, 379, 384]
    symmetry_ps = [
        pd['object'],
        pd['x_mirror'],
        pd['y_mirror'],
        pd['rotate_cw'],
        pd['rotate_ccw'],
        pd['left_half'],
        pd['right_half'], 
        pd['top_half'],
        pd['bottom_half'],
        pd['overlay'],
        pd['combine_grids_vertically'],
        pd['combine_grids_horizontally'], 
        pd['input'],
        pd['output'],
    ]

    copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
    copy_one_ps = [
            pd['objects2'],
            pd['T'], pd['F'],
            pd['input'],
            pd['rotation_invariant'],
            pd['size_invariant'],
            pd['color_invariant'],
            pd['no_invariant'],
            pd['place_into_grid'],
            pd['rows'],
            pd['columns'],
            pd['construct_mapping'],
            pd['vstack'],
            pd['hstack'],
    ]

    copy_two_tasks = [103, 55, 166, 47, 185, 398, 102, 297, 352, 372]
    copy_two_ps = [
        pd['objects2'],
        pd['T'], pd['F'],
        pd['input'],
        pd['rotation_invariant'],
        pd['size_invariant'],
        pd['color_invariant'],
        pd['no_invariant'],
        pd['construct_mapping2'],
        pd['construct_mapping3'],
        pd['area'],
        pd['has_y_symmetry'],
        pd['list_length'],
        pd['filter_list'],
        pd['contains_color'],
        pd['color2'],
    ]

    inflate_tasks = [0, 194, 216, 222, 268, 288, 306, 383]
    inflate_ps = [
            pd['input'],
            pd['object'],
            pd['output'],
            pd['area'],
            pd['kronecker'],
            pd['inflate'],
            pd['deflate'],
            pd['2'],
            pd['3'],
            pd['num_colors'],
    ]

    tasks = [get_arc_task(i) for i in inflate_tasks + copy_one_tasks + copy_two_tasks + symmetry_tasks]
    # get rid of duplicates
    primitives = list(set(inflate_ps + copy_one_ps + copy_two_ps + symmetry_ps))

    generate_ocaml_primitives(primitives)
    assert False


    grammar = Grammar.uniform(primitives)


    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=20, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=10, 
        recognitionTimeout=60, 
        featureExtractor=ArcNet,
        auxiliary=True,
        contextual=True,
        helmholtzRatio=ratio,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=5,
        resume=resume,
        reuseRecognition=True, 
        )

    training = tasks

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))



def test_recognition():
    symmetry_tasks = [30, 38, 86, 112, 115, 139, 149, 154, 163, 171, 176, 178, 209, 240, 248, 310, 359, 379, 384]
    symmetry_ps = [
        pd['object'],
        pd['x_mirror'],
        pd['y_mirror'],
        pd['rotate_cw'],
        pd['rotate_ccw'],
        pd['left_half'],
        pd['right_half'], 
        pd['top_half'],
        pd['bottom_half'],
        pd['overlay'],
        pd['combine_grids_vertically'],
        pd['combine_grids_horizontally'], 
        pd['input'],
        pd['output'],
    ]

    copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
    copy_one_ps = [
            pd['objects2'],
            pd['T'], pd['F'],
            pd['input'],
            pd['rotation_invariant'],
            pd['size_invariant'],
            pd['color_invariant'],
            pd['no_invariant'],
            pd['place_into_grid'],
            pd['rows'],
            pd['columns'],
            pd['construct_mapping'],
            pd['vstack'],
            pd['hstack'],
    ]

    copy_two_tasks = [103, 55, 166, 47, 185, 398, 102, 297, 352, 372]
    copy_two_ps = [
        pd['objects2'],
        pd['T'], pd['F'],
        pd['input'],
        pd['rotation_invariant'],
        pd['size_invariant'],
        pd['color_invariant'],
        pd['no_invariant'],
        pd['construct_mapping2'],
        pd['construct_mapping3'],
        pd['area'],
        pd['has_y_symmetry'],
        pd['list_length'],
        pd['filter_list'],
        pd['contains_color'],
        pd['color2'],
    ]

    inflate_tasks = [0, 194, 216, 222, 268, 288, 306, 383]
    inflate_ps = [
            pd['input'],
            pd['object'],
            pd['output'],
            pd['area'],
            pd['kronecker'],
            pd['inflate'],
            pd['deflate'],
            pd['2'],
            pd['3'],
            pd['num_colors'],
    ]

    primitives = inflate_ps + copy_one_ps + copy_two_ps + symmetry_ps
    primitives = list(set(primitives))

    p_dict = {str(p): i+1 for i, p in enumerate(primitives)}
    print(len(p_dict))

    num_input_ps = max(len(i) for i in [symmetry_ps, copy_one_ps,
        copy_two_ps, inflate_ps])

    def augmented_tasks(task_numbers, primitive_list, task_type):
        tasks = [get_arc_task(i) for i in task_numbers]
        for t in tasks:
            l = [0] * num_input_ps
            l[0:len(primitive_list)] = [p_dict[str(p)] for p in primitive_list]
            t.primitives = l
            t.p_dict = p_dict
            t.task_type = task_type

        return tasks


    all_tasks = []
    for tasks, ps, tt in [(symmetry_tasks, symmetry_ps, 0),
            (copy_one_tasks, copy_one_ps, 1),
            (copy_two_tasks, copy_two_ps, 2),
            (inflate_tasks, inflate_ps, 3)]:
        all_tasks += augmented_tasks(tasks, ps, tt)

    grammar = Grammar.uniform(primitives)


    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=10, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        recognitionTimeout=120, 
        # featureExtractor=TestRecNet,
        auxiliary=True,
        contextual=True,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        helmholtzRatio=0.,
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=5
        )

    training = all_tasks

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

    # model = result.recognitionModel
    # grammar = model.grammarOfTask(task)
    # grammar = grammar.untorch()
    # print('grammar: {}'.format(grammar))




def test():
    # test_rec1(0.0)
    # test_recognition()
    # test_construct_mapping()
    test_construct_mapping2()
    # test_inflate()
    # test_symmetry()

