import json
import os
import numpy as np
from numpy.random import default_rng
import random
import math

import torch
import torch.nn.functional as F

from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.domains.arc.arcPrimitives import Grid, Object, _get, _color, _objects, tgrid, tobject, tcolor

def load_task(task_id, task_path='data/ARC/data/training/'):
    filename = task_path + task_id + '.json'

    with open(filename, 'r') as f:
        task_dict = json.load(f)

    task_dict['name'] = task_id
    return task_dict

def make_tasks_getobjectcolor():

    # ---------------------------------------------
    # TASK that gets 3rd object
    # ---------------------------------------------
    
    array0_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array0_out = [[0, 0, 2], 
                 [0, 0, 2], 
                 [0, 0, 2]]
    arc0_in = Grid(array0_in)
    arc0_out = Object(array0_out)
    should_be = _get(_objects(arc0_in))(2)
    assert arc0_out == should_be, 'incorrect example created'

    


    array1_in = [[3, 3, 3], 
                 [9, 9, 9], 
                 [6, 6, 6]]
    array1_out = [[0, 0, 0], 
                 [0, 0, 0], 
                 [6, 6, 6]]
    arc1_in = Grid(array1_in)
    arc1_out = Object(array1_out)
    should_be = _get(_objects(arc1_in))(2)
    assert arc1_out == should_be, 'incorrect example created'

    example = ((arc0_in,), arc0_out)
    examples0 = [((arc0_in,), arc0_out), ((arc1_in,), arc1_out)]
    task_getobject = Task(
            " GET_3rd_OBJECT",
            arrow(tgrid, tobject),
            examples0,
        )


    # ---------------------------------------------
    # TASK that gets 3rd object's color
    # ---------------------------------------------
    
    
    array0_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array0_out = 2
    arc0_in = Grid(array0_in)
    arc0_out = array0_out
    should_be = _color(_get(_objects(arc0_in))(2)) 
    assert arc0_out == should_be, 'incorrect example created'

    


    array1_in = [[3, 3, 3], 
                 [9, 9, 9], 
                 [6, 6, 6]]
    array1_out = 6
    arc1_in = Grid(array1_in)
    arc1_out = array1_out
    should_be = _color(_get(_objects(arc1_in))(2)) 
    assert arc1_out == should_be, 'incorrect example created'

    example = ((arc0_in,), arc0_out)
    examples0 = [((arc0_in,), arc0_out), ((arc1_in,), arc1_out)]
    task_getobjectcolor = Task(
            " GET_3rd_OBJECT_COLOR",
            arrow(tgrid, tcolor),
            examples0,
        )

    # ---------------------------------------------
    # PRINT
    # ---------------------------------------------
    training= [task_getobject, task_getobjectcolor]
    testing = []

    return training, testing


def make_arc_task(task_id):
    d = load_task(task_id)
    
    examples = [((ArcInput(ex["input"], d["train"]),), 
        Grid(ex["output"])) for ex in d["train"]]
    # include test examples, but their output is not included in input
    examples += [((ArcInput(ex["input"], d["train"]),),
        Grid(ex["output"])) for ex in d["test"]]

    task = Task(task_id, 
            arrow(tinput, tgrid),
            examples)
    return task

def arc_tasks(task_ids):
    return [make_arc_task(task_id) for task_id in task_ids]

def full_arc_task(include_eval=False):
    training_dir = 'data/ARC/data/training/'
    evaluation_dir = 'data/ARC/data/evaludation/'

    # take off last five chars of name to get rid of '.json'
    task_ids = [t[:-5] for t in os.listdir(training_dir)]
    if include_eval:
        task_ids += [t[:-5] for t in os.listdir(evaluation_dir)]

    return arc_tasks(task_ids)

