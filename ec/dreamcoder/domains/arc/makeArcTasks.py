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

    examples = [((ArcExample(np.array(ex["input"])),), 
            ArcExample(np.array(ex["output"])))
            for ex in d["train"]]

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

def num_pixels():
    return 3

def make_tasks2():
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

