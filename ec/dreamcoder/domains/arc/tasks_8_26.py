from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.makeTasks import get_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcInput import load_task

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
        if task == 96:
            objects = p._objects2(p._input(i))(True)(False)
            objects = p._filter_list(objects)(lambda o: p._not_pixel(o))
            return p._place_into_grid(objects)
        elif task == 80:
            obj2 = lambda g: p._objects2(g)(False)(False)
            objs = p._construct_mapping(obj2)(obj2)('rotation')(i)
            return p._place_into_grid(objs)
        elif task == 81:
            obj2 = lambda g: p._objects2(g)(True)(True)
            objs = p._construct_mapping(obj2)(obj2)('color')(i)
            return p._place_into_grid(objs)
        elif task == 261:
            obj2 = lambda g: p._rows(g)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 72:
            obj2 = lambda g: p._columns(g)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 55:
            return p._construct_mapping2(lambda i: i)(lambda i: i)('color')(i)
        elif task == 103:
            return p._construct_mapping2(lambda i: i)(lambda i: i)('rotation')(i)
        elif task == 330:
            obj2 = lambda g: p._objects2(g)(False)(False)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 229:
            obj2 = lambda g: p._objects2(g)(True)(False)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 168:
            objs = lambda g: p._objects2(g)(False)(False)
            objs = p._construct_mapping3(objs)(i)(
                    lambda o: p._area(o))(
                    lambda o: p._color(o))(
                    lambda o: lambda c: p._color_in(o)(c))
            return p._place_into_grid(objs)
        elif task == 185:
            return p._construct_mapping4(i)(lambda g: p._area(g))
        elif task == 47:
            def good(g):
                objs = p._objects2(g)(False)(False)
                # print('objs: {}'.format(objs))
                red_objs = p._filter_list(objs)(lambda o: p._contains_color(o)(2))
                # print('red_objs: {}'.format(red_objs))
                good = p._eq(p._length(red_objs))(1)
                # print('good: {}'.format(good))
                return good
            return p._construct_mapping4(i)(good)


    return prog

def run():
    for i in range(400):
        check_solves(get_arc_task(i), program(i))

