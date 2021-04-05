import datetime
import os
import random

import binutil # needed for importing things properly iirc

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
from dreamcoder.domains.arc.main import ArcNet

from dreamcoder.domains.arc.arcPrimitives import primitive_dict as p
from dreamcoder.domains.arc.arcPrimitives import generate_ocaml_primitives
from dreamcoder.domains.arc.test import test

def symmetry_experiment():
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
    training = [get_arc_task(i) for i in symmetry_tasks]

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/',
                           **args)

    # run the DreamCoder learning process for the set number of iterations
    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))

def rectangles():
    ps = [
        p['input'],
        p['object'],
        p['fill_rectangle'],
        p['overlay'],
        p['place_into_grid'],
        p['list_of_one'],
        p['shell'],
        p['objects2'], p['T'], p['F'],
        p['hollow'],
    ]

    ps += [p['color' + str(i)] for i in range(0, MAX_COLOR+1)]

    # get rid of duplicates
    primitives = ps
    tasks = [get_arc_task(i) for i in range(400)]

    # generate_ocaml_primitives(primitives)
    # assert False

    grammar = Grammar.uniform(primitives)

    args = commandlineArguments(
        enumerationTimeout=120, 
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        recognitionTimeout=3600, 
        # featureExtractor=ArcNet,
        # auxiliary=True, # train our feature extractor too
        # contextual=True, # use bi-gram model, not unigram
        a=3,  # max arity of compressed primitives
        maximumFrontier=5, # number of programs used for compression
        topK=2, 
        pseudoCounts=30.0,
        helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=15,
        )

    generator = ecIterator(grammar,
                           tasks,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/' + str(datetime.date.today()),
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))





def misc():
    ps = [
        p['map_i_to_j'],
        p['list_of_one'],
        p['place_into_grid'],
        p['place_into_input_grid'],
        p['sort_incr'],
        p['sort_decr'],
        p['color_in'],
        p['color'],
        p['overlay'],
        p['object'],
        p['objects2'],
        p['T'],
        p['F'],
        p['hblock'],
        p['area'],
        p['input'],
        p['move_down2'],
        p['get_first'],
    ]

    ps += [p['color' + str(i)] for i in range(0, MAX_COLOR+1)]
    ps += [p[str(i)] for i in range(0, 10)]

    # get rid of duplicates
    primitives = ps
    tasks = [get_arc_task(i, use_toutput=False) for i in range(400)]

    # generate_ocaml_primitives(primitives)
    # assert False

    grammar = Grammar.uniform(primitives)

    args = commandlineArguments(
        enumerationTimeout=120, 
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        recognitionTimeout=3600, 
        # featureExtractor=ArcNet,
        # auxiliary=True, # train our feature extractor too
        # contextual=True, # use bi-gram model, not unigram
        a=3,  # max arity of compressed primitives
        maximumFrontier=5, # number of programs used for compression
        topK=2, 
        pseudoCounts=30.0,
        helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=15,
        )

    generator = ecIterator(grammar,
                           tasks,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/' + str(datetime.date.today()),
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))


def main():
    color_ps = [p['color'+str(i)] for i in range(0,10)]
    pixelwise_ps = [
        p['complement'],
        p['return_subgrids'],
        p['stack_xor'],
        p['stack_and'],
        p['grid_split'],
        p['color_in'],
        p['stack_overlay'],
        p['input'] ]
    symmetry_ps = [
        p['object'],
        p['x_mirror'],
        p['y_mirror'],
        p['rotate_cw'],
        p['rotate_ccw'],
        p['left_half'],
        p['right_half'], 
        p['top_half'],
        p['bottom_half'],
        p['overlay'],
        p['combine_grids_vertically'],
        p['combine_grids_horizontally'], 
        p['input'],
    ]

    copy_one_ps = [
            p['objects2'],
            p['T'], p['F'],
            p['input'],
            p['rotation_invariant'],
            p['size_invariant'],
            p['color_invariant'],
            p['no_invariant'],
            p['place_into_grid'],
            p['rows'],
            p['columns'],
            p['construct_mapping'],
            p['vstack'],
            p['hstack'],
    ]

    copy_two_ps = [
        p['objects2'],
        p['T'], p['F'],
        p['input'],
        p['rotation_invariant'],
        p['size_invariant'],
        p['color_invariant'],
        p['no_invariant'],
        p['construct_mapping2'],
        p['construct_mapping3'],
        p['area'],
        p['has_y_symmetry'],
        p['list_length'],
        p['filter_list'],
        p['contains_color'],
        p['color2'],
    ]

    inflate_ps = [
            p['input'],
            p['object'],
            p['area'],
            p['kronecker'],
            p['inflate'],
            p['deflate'],
            p['2'],
            p['3'],
            p['num_colors'],
    ]

    misc_ps = [
        p['map_i_to_j'],
        p['list_of_one'],
        p['place_into_grid'],
        p['place_into_input_grid'],
        p['sort_incr'],
        p['sort_decr'],
        p['color_in'],
        p['color'],
        p['overlay'],
        p['object'],
        p['objects2'],
        p['T'],
        p['F'],
        p['hblock'],
        p['vblock'],
        p['area'],
        p['input'],
        p['move_down2'],
        p['get_first'],
        p['shell'],
        p['hollow'],
        p['fill_rectangle'],
        p['enclose_with_ring'],
        p['is_rectangle'],
        p['is_rectangle_not_pixel'],
    ]

    # get rid of duplicates
    primitives = list(set(inflate_ps + copy_one_ps + copy_two_ps + symmetry_ps + misc_ps + pixelwise_ps + color_ps ))
    #primitives = list(set(p[prim] for prim in p))
    #primitives = list(set(pixelwise_ps+color_ps))
    tasks = [get_arc_task(i) for i in range(400)]
    #tasks = get_eval_tasks()

    generate_ocaml_primitives(primitives)
    # assert False

    grammar = Grammar.uniform(primitives)

    args = commandlineArguments(
        enumerationTimeout=120, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=1, 
        recognitionTimeout=3600, 
        featureExtractor=ArcNet,
        auxiliary=True, # train our feature extractor too
        contextual=True, # use bi-gram model, not unigram
        a=3,  # max arity of compressed primitives
        maximumFrontier=5, # number of programs used for compression
        topK=2, 
        pseudoCounts=30.0,
        helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=15,
        )

    generator = ecIterator(grammar,
                           tasks,
                           testingTasks=[],
                           outputPrefix='./experimentOutputs/arc/' + str(datetime.date.today()),
                           **args)

    for i, result in enumerate(generator):
        print('ecIterator count {}'.format(i))


def tasks():
    # some common task values for reference
    training = [get_arc_task(i) for i in range(0, 400)] 
    copy_one_tasks = [11, 14, 15, 80, 81, 94, 159, 281, 316, 330, 72, 261, 301, 234]
    copy_two_tasks = [103, 55, 166, 47, 185, 398, 102, 297, 352, 372, 333]
    symmetry_tasks = [30, 38, 52, 56, 66, 70, 82, 86, 105, 108, 112, 115, 116, 139, 141, 149, 151, 154, 163, 171, 176, 178, 179, 209, 210, 240, 241, 243, 248, 310, 346, 359, 360, 379, 371, 384]
    # I think these are just the ones that we have solved.
    symmetry_tasks = [11, 14, 15, 80, 81, 85, 159, 261, 281, 301, 373, 30, 154, 178, 240, 86, 139, 379, 149, 112, 384, 115, 171, 209, 176, 38, 359, 248, 163, 310, 82, 141, 151]
    inflate_tasks = [0, 194, 216, 222, 268, 288, 306, 383]
    pixelwise_tasks = [346,226,394,71,5,25,143,235,256,317,320,371,385]
#test()
#check_tasks()
# generate_ocaml_primitives()
#assert False
main()
# misc()
# rectangles()
