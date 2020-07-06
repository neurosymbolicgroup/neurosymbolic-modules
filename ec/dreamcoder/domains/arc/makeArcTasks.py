import json
import os
import numpy as np
from numpy.random import default_rng

from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives, ArcExample, _gridempty

def load_task(task_id, task_path='data/ARC/data/training/'):
    filename = task_path + task_id + '.json'

    with open(filename, 'r') as f:
        task_dict = json.load(f)

    task_dict['name'] = task_id
    return task_dict

def make_task(task_id):
    d = load_task(task_id)
    # ---------------------------------------------
    # TASK where input grid is same as output grid
    # ---------------------------------------------
    
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["input"]))
            for training_example in d["train"]]

    task_identity = Task(
            task_name + "IDENTITY",
            arrow(tgrid, tgrid),
            examples,
            make_features(examples)
        )

    i, o = examples[0]
    i = i[0] # only one input arg
    # print('i,: {}'.format(i,))
    # print('o: {}'.format(o))

    expected = i.map_i_to_j(3, 4).map_i_to_j(1, 5).map_i_to_j(2, 6) 
    assert o == expected, "not good"

    task = Task(task_id, 
            arrow(tgrid, tgrid),
            examples,
            make_features(examples))
    return task

def make_tasks():
    task_name = "0d3d703e"
    task = make_task(task_name)
    training = [task]
    testing = []
    return training, testing

def make_tasks2():
    # ---------------------------------------------
    # TASK that takes in grid and outputs blank grid of same shape as INPUT
    # ---------------------------------------------
    
    task_blank_in = Task(task_name + "BLANK_IN",
            arrow(tgrid, tgrid),
            examples,
            make_features(examples)
        )

    # ---------------------------------------------
    # TASK that maps 2 colors
    # ---------------------------------------------
    
    array1_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array1_out = [[4, 5, 2], 
                  [4, 5, 2], 
                  [4, 5, 2]]
    arc1_in = ArcExample(array1_in)
    arc1_out = ArcExample(array1_out)
    should_be = arc1_in.map_i_to_j(3, 4).map_i_to_j(1, 5)#.map_i_to_j(2, 3)
    assert arc1_out == should_be, 'incorrect example created'

    example = (arc1_in,), arc1_out
    examples1 = [example]

    task_1 = Task(
            task_name + "2_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples1,
            features=make_features(examples)
        )

    # ---------------------------------------------
    # TASK that maps 3 colors
    # ---------------------------------------------
    
    array2_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array2_out = [[4, 5, 3], 
                  [4, 5, 3], 
                  [4, 5, 3]]
    arc2_in = ArcExample(array2_in)
    arc2_out = ArcExample(array2_out)
    should_be = arc2_in.map_i_to_j(3, 4).map_i_to_j(1, 5).map_i_to_j(2, 3)
    assert arc2_out == should_be, 'incorrect example created'

    example = (arc2_in,), arc2_out
    examples2 = [example]

    # ex: ((arc1_in,), arc1_out), tuple of length one?
    # ex[0]: 

    

    task_2 = Task(
            task_name + "3_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples2,
            features=make_features(examples)
        )

    # ---------------------------------------------
    # TASK that maps 1 colors
    # ---------------------------------------------
    
    array0_in = [[3, 1, 1], 
                 [3, 1, 1], 
                 [3, 1, 1]]
    array0_out = [[4, 1, 1], 
                  [4, 1, 1], 
                  [4, 1, 1]]
    arc0_in = ArcExample(array0_in)
    arc0_out = ArcExample(array0_out)

    example = (arc0_in,), arc0_out
    examples0 = [example]

    # ex: ((arc1_in,), arc1_out), tuple of length one?
    # ex[0]: 

    task_0 = Task(
            task_name + "1_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples0,
            features=make_features(examples0)
        )

    # ---------------------------------------------
    # LIST OF ALL TASKS
    # ---------------------------------------------
    

    # print(task_1.examples)
    # print(task_2.examples)

    training = [task_0]
    testing = []
    return training_testing


def num_pixels():
    return 3

def make_tasks3():
    print('making tasks')
    training = []
    testing = []

    num_tasks = 10
    n = num_pixels()
    rng = default_rng()

    for i in range(num_tasks):
        worked = False
        while not worked:
            inp = rng.choice(range(0, 11), size=n, replace=False)
            out = rng.choice(range(0, 11), size=n, replace=False)
            examples = [((ArcExample(inp),),
                ArcExample(out))]
            i, o = examples[0]
            i = i[0] # only one input arg
            expected = i
            for (c1, c2) in zip(inp, out):
                expected = expected.map_i_to_j(c1, c2)

            worked = o == expected

        t = Task(str((inp, out)), 
                arrow(tgrid, tgrid),
                examples,
                make_features(examples))
        training.append(t)

    print('done making tasks')
    return training, testing



def make_features(examples):
    # concatenate all of the input/output grids into one flat array

    # [ ( (arc1_in,), arc1_out) ]
    examples = [np.concatenate((ex[0][0].grid, ex[1].grid)) for ex in examples]
    features = np.concatenate(examples).flatten()
    return features


def run_stuff():
    d = load_task('0d3d703e')
    print(d['train'][0])
    print(d['train'][1])
    print(d['train'])
    print(d['test'])

