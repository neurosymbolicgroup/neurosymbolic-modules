import json
import os
import numpy as np
from numpy.random import default_rng
import random
import math

from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives, ArcExample, ArcObject, _gridempty, ArcInput, tinput

def load_task(task_id, task_path='data/ARC/data/training/'):
    filename = task_path + task_id + '.json'

    with open(filename, 'r') as f:
        task_dict = json.load(f)

    task_dict['name'] = task_id
    return task_dict

def make_andy_task():
    array1_in = [[3, 1, 2]] 
    array1_out = [[4, 5, 3]]
    arc1_in = ArcExample(array1_in)
    arc1_out = ArcExample(array1_out)
    should_be = arc1_in.map_i_to_j(3, 4).map_i_to_j(1, 5).map_i_to_j(2, 3)
    assert arc1_out == should_be, 'incorrect example created'

    example = (arc1_in,), arc1_out
    examples = [example]

    task = Task(
            "3_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples,
            features=make_features(examples)
        )
    return task

def make_andy_task2():
    array1_in = [[3, 1, 2],
                 [3, 1, 2],
                 [3, 1, 2]] 
    array1_out = [[4, 5, 3],
                  [4, 5, 3],
                  [4, 5, 3]]
    arc1_in = ArcExample(array1_in)
    arc1_out = ArcExample(array1_out)
    should_be = arc1_in.map_i_to_j(3, 4).map_i_to_j(1, 5).map_i_to_j(2, 3)
    assert arc1_out == should_be, 'incorrect example created'

    example = (arc1_in,), arc1_out
    examples = [example]

    task = Task(
            "3_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples,
            features=make_features(examples)
        )
    return task

def get_tasks():
    return [make_map_arcinput_task()], []

def robustfill_task(num_colors=7):
    d = {1: 5, 2: 6, 3: 4, 4: 3, 5: 1, 6: 2, 8: 9, 9: 8}
    grid = np.array([[1,2,3],[1,2,3],[1,2,3]])

    def sample(i, j, k):
        return np.array([[i,j,k],[i,j,k],[i,j,k]])

    inp = 0
    def next_input():
        nonlocal inp
        inp += 1
        if inp == num_colors + 1:
            inp = 1
        return inp

    num_examples = math.ceil(num_colors/3)
    examples = []
    for i in range(num_examples):
        a, b, c = next_input(), next_input(), next_input()
        input_grid = ArcExample(sample(a, b, c))
        output_grid = input_grid.transform(d)
        examples.append((input_grid, output_grid))

    return examples


    



def identity():
    task_name = "d07ae81c"
    d = load_task(task_name)

    # ---------------------------------------------
    # TASK where input grid is same as output grid
    # ---------------------------------------------
    
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["input"]))
            for training_example in d["train"]]

    examples = [examples[0]]

    task_identity = Task(
            task_name + "IDENTITY",
            arrow(tgrid, tgrid),
            examples,
            make_features(examples)
        )

    return task_identity


def make_map_task():
    task_id = '0d3d703e'
    d = load_task(task_id)
    
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["output"]))
            for training_example in d["train"]]

    examples = examples[0:1]
    print('examples: {}'.format(examples))
    i, o = examples[0]
    i = i[0] # only one input arg
    print('i,: {}'.format(i,))
    print('o: {}'.format(o))

    expected = i.map_i_to_j(3, 4).map_i_to_j(1, 5)
    assert o == expected, "not good: {}, {}".format(o, expected)
    # expected = transform_fn(3)(i)(4)(6)(5)
    # assert o == expected, "not good: {}, {}".format(o, expected)

    task = Task(task_id, 
            arrow(tgrid, tgrid),
            examples,
            make_features(examples))
    return task

def make_task(task_id):
    d = load_task(task_id)
    
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["output"]))
            for training_example in d["train"]]

    task = Task(task_id, 
            arrow(tgrid, tgrid),
            examples)
            # make_features(examples))
    return task

def make_tasks_anshula():
    task_name = "0d3d703e" # 3 stripes mapping
    d = load_task(task_name)

    task_name = "85c4e7cd" # concentric circles
    d2 = load_task(task_name)

    # ---------------------------------------------
    # TASK where input grid is same as output grid
    # ---------------------------------------------
    
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["input"]))
            for training_example in d["train"]] + \
             [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["input"]))
            for training_example in d2["train"]]

    task_identity = Task(
            task_name + " IDENTITY",
            arrow(tgrid, tgrid),
            examples,
            # make_features(examples)
        )

    # ---------------------------------------------
    # TASK that takes in grid and outputs blank grid of same shape as INPUT
    # ---------------------------------------------
    examples = [((ArcExample(training_example["input"]),),
            _gridempty(ArcExample(training_example["input"])))
            for training_example in d["train"]]
   
    task_blank_in = Task(task_name + " BLANK_IN",
            arrow(tgrid, tgrid),
            examples,
            # make_features(examples)
        )



    # ---------------------------------------------
    # TASK that gets 3rd object
    # ---------------------------------------------
    
    array0_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array0_out = [[3, 0, 0], 
                 [3, 0, 0], 
                 [3, 0, 0]]
    arc0_in = ArcExample(array0_in)
    arc0_out = ArcObject(array0_out)
    should_be = arc0_in.get_objects().get(2) # gets objects in color order, so object with color 3 is in 3rd position
    assert arc0_out == should_be, 'incorrect example created'

    


    array1_in = [[0, 3, 4], 
                 [5, 6, 7], 
                 [9, 8, 7]]
    array1_out = [[0, 0, 4], 
                 [0, 0, 0], 
                 [0, 0, 0]]
    arc1_in = ArcExample(array1_in)
    arc1_out = ArcObject(array1_out)
    should_be = arc1_in.get_objects().get(2) # gets objects in color order, so object with color 3 is in 3rd position
    assert arc1_out == should_be, 'incorrect example created'
    

    array2_in = [[0, 1, 2], 
                 [3, 4, 5], 
                 [6, 7, 8]]
    array2_out = [[0, 0, 2], 
                 [0, 0, 0], 
                 [0, 0, 0]]
    arc2_in = ArcExample(array2_in)
    arc2_out = ArcObject(array2_out)
    should_be = arc2_in.get_objects().get(2) # gets objects in color order, so object with color 3 is in 3rd position
    assert arc2_out == should_be, 'incorrect example created'
    

    example = ((arc0_in,), arc0_out)
    examples0 = [((arc0_in,), arc0_out), ((arc1_in,), arc1_out), ((arc1_in,), arc1_out)]
    task_getobject = Task(
            task_name + " GET_3rd_OBJECT",
            arrow(tgrid, tgrid),
            examples0,
            features=make_features(examples0)
        )


    # ---------------------------------------------
    # TASK that gets 3rd object's color
    # ---------------------------------------------
    
    array0_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array0_out = 3
    arc0_in = ArcExample(array0_in)
    arc0_out = array0_out
    should_be = arc0_in.get_objects().get(2).get_color() # gets objects in color order, so object with color 3 is in 3rd position
    assert arc0_out == should_be, 'incorrect example created'


    array1_in = [[0, 3, 4], 
                 [5, 6, 7], 
                 [9, 8, 7]]
    array1_out = 4
    arc1_in = ArcExample(array1_in)
    arc1_out = array1_out
    should_be = arc1_in.get_objects().get(2).get_color() # gets objects in color order, so object with color 3 is in 3rd position
    assert arc1_out == should_be, 'incorrect example created'

    array2_in = [[0, 1, 2], 
                 [3, 4, 5], 
                 [6, 7, 8]]
    array2_out = 2
    arc2_in = ArcExample(array2_in)
    arc2_out = array2_out
    should_be = arc2_in.get_objects().get(2).get_color() # gets objects in color order, so object with color 3 is in 3rd position
    assert arc2_out == should_be, 'incorrect example created'

    array3_in = [[1, 0, 7], 
                 [1, 0, 7], 
                 [1, 0, 7]]
    array3_out = 7
    arc3_in = ArcExample(array3_in)
    arc3_out = array3_out
    should_be = arc3_in.get_objects().get(2).get_color() # gets objects in color order, so object with color 3 is in 3rd position
    assert arc3_out == should_be, 'incorrect example created'


    example = ((arc0_in,), arc0_out)
    examples0 = [((arc0_in,), arc0_out), ((arc1_in,), arc1_out), ((arc2_in,), arc2_out), ((arc3_in,), arc3_out)]
    task_getcolor = Task(
            task_name + " GET_3rd_OBJECT_COLOR",
            arrow(tgrid, tint),
            examples0,
            # features=make_features(examples0)
        )

    # ---------------------------------------------
    # TASK that maps an object color to another object color
    # ---------------------------------------------
    
    # should always map 
        # the color of the 1st object (one with lowest color of the three) 
        # to the color of the 3rd object (one with the highest color of the three)

    array0_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array0_out = [[3, 3, 2], 
                 [3, 3, 2], 
                 [3, 3, 2]]
    arc0_in = ArcExample(array0_in)
    arc0_out = ArcExample(array0_out)
    should_be = arc0_in.map_i_to_j(arc0_in.get_objects().get(0).get_color(), arc0_in.get_objects().get(2).get_color())
    assert arc0_out == should_be, 'incorrect example created'
    
    example0 = (arc0_in,), arc0_out

    array1_in = [[5, 4, 6], 
                 [5, 4, 6], 
                 [5, 4, 6]]
    array1_out = [[5, 6, 6], 
                 [5, 6, 6], 
                 [5, 6, 6]]
    arc1_in = ArcExample(array1_in)
    arc1_out = ArcExample(array1_out)
    should_be = arc1_in.map_i_to_j(arc1_in.get_objects().get(0).get_color(), arc1_in.get_objects().get(2).get_color())
    assert arc1_out == should_be, 'incorrect example created'
    example1 = (arc1_in,), arc1_out

    examples = [example0, example1]

    # ex: ((arc1_in,), arc1_out), tuple of length one?
    # ex[0]: 

    

    # task_mapobjectcolor = Task(
    #         task_name + " MAP_OBJECT_COLOR",
    #         arrow(tgrid, tgrid),
    #         examples,
    #         features=make_features(examples0)
    #     )

    # ---------------------------------------------
    # TASK that maps 1 colors
    # ---------------------------------------------
    
    array0_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array0_out = [[4, 1, 2], 
                 [4, 1, 2], 
                 [4, 1, 2]]
    arc0_in = ArcExample(array0_in)
    arc0_out = ArcExample(array0_out)
    should_be = arc0_in.map_i_to_j(3, 4)
    assert arc0_out == should_be, 'incorrect example created'

    example = (arc0_in,), arc0_out
    examples0 = [example]

    # ex: ((arc1_in,), arc1_out), tuple of length one?
    # ex[0]: 

    

    task_0 = Task(
            task_name + " 1_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples0,
            features=make_features(examples0)
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
            task_name + " 2_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples1,
            features=make_features(examples1)
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
            task_name + " 3_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples2,
            features=make_features(examples2)
        )

    # ---------------------------------------------
    # PRINT
    # ---------------------------------------------
    

    # training = [task_identity, task_blank_in, task_1]
    # testing = [task_identity]

    training = [task_identity, task_blank_in, task_getobject, task_getcolor, task_0 ] #task_mapobjectcolor
    testing = [task_1, task_2]

    return training, testing


def num_pixels():
    return 3

def make_random_map_tasks():
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

def make_map_arcinput_task():
    task_id = '0d3d703e'
    d = load_task(task_id)["train"]
    
    examples = [((ArcInput(ex["input"], d),), 
        ArcExample(ex["output"])) for ex in d]

    # print('examples: {}'.format(examples[0]))

    i, o = examples[0]
    i = i[0] # only one input arg
    # print('i,: {}'.format(i,))
    # print('o: {}'.format(o))
    expected = i.for_each_color(lambda grid: lambda c: grid.map_i_to_j(c, 
        i.get_output_grids().color_at_location2(i.get_input_grids().location2_with_color(c))))
    assert o == expected, "not good: {}, {}".format(o, expected)

    task = Task(task_id,
            arrow(tinput, tgrid),
            examples)

    return task


# def make_features(examples):
#     # [ ( (arc1_in,), arc1_out) ]
#     features = []
#     for ex in examples:
#         inp, outp = ex[0][0], ex[1]
#         features.append(inp.grid)
#         features.append(outp.grid)
    
#     return features

def make_features(examples):
    # concatenate all of the input/output grids into one flat array

    # [ ( (arc1_in,), arc1_out) ]
    examples = [np.concatenate((ex[0][0].grid, ex[1].grid)) for ex in examples]
    features = np.concatenate(examples).flatten()
    return features


def full_arc_task(include_eval=False):
    training_dir = 'data/ARC/data/training/'
    evaluation_dir = 'data/ARC/data/evaludation/'

    # take off last five chars of name to get rid of '.json'
    task_ids = [t[:-5] for t in os.listdir(training_dir)]
    if include_eval:
        task_ids += [t[:-5] for t in os.listdir(evaluation_dir)]

    tasks = [make_task(task_id) for task_id in task_ids]
    tasks = sorted(tasks, key=lambda t: len(t.examples))
    print('tasks: {}'.format([t.name for t in tasks]))
    n = [len(t.examples) for t in tasks]
    n2 = np.unique(n, return_counts=True)
    print('n2: {}'.format(n2))
    assert False
    return tasks



