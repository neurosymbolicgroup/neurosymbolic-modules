from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.makeTasks import get_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcInput import load_task
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as pd
from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.domains.arc.main import ArcNet, TestRecNet, TestRecNet2

def check_solves(task, program):
    for i, ex in enumerate(task.examples):
        inp, out = ex[0][0], ex[1]
        predicted = program(inp)
        if predicted is None:
            return

        if predicted != out:
            print('didnt solve: {}'.format(task.name))
            print('Failed example ' + str(i) + ': input=')
            print(p._input(inp))
            print('output=')
            print(out)
            print('predicted=')
            print(predicted)
            # assert False, 'did NOT pass!'
            print('Did not pass')
            return
    print('Passed {}'.format(task.name))

def program(task):
    def prog(i):
        if task == 0:
            return p._kronecker(p._input(i))(p._input(i))
        elif task == 47:
            def good(g):
                objs = p._objects2(g)(False)(False)
                # print('objs: {}'.format(objs))
                red_objs = p._filter_list(objs)(lambda o: p._contains_color(o)(2))
                # print('red_objs: {}'.format(red_objs))
                good = p._length(red_objs)
                # print('good: {}'.format(good))
                return good
            return p._construct_mapping3(good)(i)
        elif task == 48:
            return p._get_first(p._sort_incr(p._objects2(p._input(i))(False)(True))(lambda obj: p._area(obj)))
        elif task == 52:
            return p._place_into_grid(p._list_of_one(p._move_down2(p._input(i))))(i)
        elif task == 55:
            obj2 = lambda g: p._list_of_one(g)
            objs = p._construct_mapping(obj2)(obj2)('color')(i)
            return p._get_first(objs)
        elif task == 72:
            obj2 = lambda g: p._columns(g)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)(i)
        elif task == 80:
            obj2 = lambda g: p._objects2(g)(False)(False)
            objs = p._construct_mapping(obj2)(obj2)('rotation')(i)
            return p._place_into_grid(objs)(i)
        elif task == 81:
            obj2 = lambda g: p._objects2(g)(True)(True)
            objs = p._construct_mapping(obj2)(obj2)('color')(i)
            return p._place_into_grid(objs)(i)
        elif task == 96:
            objects = p._objects2(p._input(i))(True)(False)
            objects = p._filter_list(objects)(lambda o: p._not_pixel(o))
            return p._place_into_grid(objects)(i)
        elif task == 97:
            objects = p._objects2(p._input(i))(True)(False)
            objects = p._map(lambda o: p._shell(o))(objects)
            return p._place_into_grid(objects)(i)
        elif task == 102:
            return p._construct_mapping3(lambda g: p._has_y_symmetry(g))(i)
        elif task == 103:
            return p._construct_mapping2('rotation')(i)
        elif task == 119:
            return p._place_into_grid(p._map(lambda o: p._overlay(p._color_in(p._hollow(o))(8))(o))(p._objects2(p._input(i))(True)(True)))(i)
        elif task == 128:
            return p._color_in(p._map_i_to_j(p._input(i))(0)(1))(p._color(p._input(i)))
        elif task == 138:
            objects = p._objects2(p._input(i))(True)(True)
            objects = p._map(lambda o: p._overlay(o)(p._fill_rectangle(o)(7)))(objects)
            return p._place_into_grid(objects)(i)
        elif task == 165:
            o1 = p._object(p._input(i))
            o2 = p._fill_rectangle(o1)(2)
            o3 = p._overlay(o1)(o2)
            return p._place_into_grid(p._list_of_one(o3))(i)
        elif task == 170:
            return p._shell(p._fill_rectangle(p._input(i))(8))
        elif task == 185:
            return p._construct_mapping3(lambda g: p._area(g))(i)
        elif task == 194:
            return p._kronecker(p._deflate_detect_scale(p._object(p._input(i))))(p._deflate_detect_scale(p._object(p._input(i))))
        elif task == 222:
            return p._inflate(p._input(i))(3)
        elif task == 229:
            obj2 = lambda g: p._objects2(g)(True)(False)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)(i)
        elif task == 234:
            obj2 = lambda g: p._objects2(g)(True)(True)
            return p._vstack(p._construct_mapping(obj2)(lambda g: p._rows(g))('none')(i))
        elif task == 261:
            obj2 = lambda g: p._rows(g)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)(i)
        elif task == 330:
            obj2 = lambda g: p._objects2(g)(False)(False)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)(i)
        elif task == 338: 
            return p._hblock(p._area(p._input(i)))(p._color(p._input(i)))
        elif task == 372:
            return p._construct_mapping2('color')(i)
        elif task == 398:
            return p._construct_mapping3(lambda g: p._area(g))(i)

    return prog


def check_tasks():
    for i in range(400):
        check_solves(get_arc_task(i), program(i))

