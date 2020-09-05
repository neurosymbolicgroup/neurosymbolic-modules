import json
import os
import numpy as np
from numpy.random import default_rng
import random
import math

from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.domains.arc.arcPrimitives import *
import dreamcoder.domains.arc.arcPrimitives as p

def make_rotation_tasks():

    # ---------------------------------------------
    # TASK that gets 3rd object
    # ---------------------------------------------
    
    array0_in = np.array(
                [[0, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])
    array0_out = np.array(
                [[0, 0, 0, 0, 0], 
                 [0, 1, 0, 0, 0], 
                 [0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 0]])
    arc0_in = Grid(array0_in)
    arc0_out = Grid(array0_out)
    should_be = p._draw_line_down(arc0_in)(p._get(p._objects(arc0_in))(0))
    assert arc0_out == should_be, 'incorrect example created'

    


    array1_in = np.array(
                [[0, 0, 0, 1, 0], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])
    array1_out = np.array(
                [[0, 0, 0, 1, 0], 
                 [0, 0, 0, 1, 0], 
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 1, 0]])
    arc1_in = Grid(array1_in)
    arc1_out = Grid(array1_out)
    should_be = p._draw_line_down(arc1_in)(p._get(p._objects(arc1_in))(0))
    assert arc1_out == should_be, 'incorrect example created'

    example = ((arc0_in,), arc0_out)
    examples0 = [((arc0_in,), arc0_out), ((arc1_in,), arc1_out)]
    task_getobject = Task(
            "drawLineDown",
            arrow(tgrid, tgrid),
            examples0,
        )


    # ---------------------------------------------
    # TASK that gets 3rd object's color
    # ---------------------------------------------
    
    
    array0_in = np.array([[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]])
    array0_out = 2
    arc0_in = Grid(array0_in)
    arc0_out = array0_out
    should_be = p._color(p._get(p._objects(arc0_in))(2)) 
    assert arc0_out == should_be, 'incorrect example created'

    


    array1_in = np.array([[3, 3, 3], 
                 [9, 9, 9], 
                 [6, 6, 6]])
    array1_out = 6
    arc1_in = Grid(array1_in)
    arc1_out = array1_out
    should_be = p._color(p._get(p._objects(arc1_in))(2)) 
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

    return training



