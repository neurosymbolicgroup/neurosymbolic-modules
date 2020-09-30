from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.makeTasks import get_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcInput import load_task
from dreamcoder.domains.arc.arcPrimitives import primitive_dict as pd
from dreamcoder.dreamcoder import commandlineArguments, ecIterator

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
        if task == 47:
            def good(g):
                objs = p._objects2(g)(False)(False)
                # print('objs: {}'.format(objs))
                red_objs = p._filter_list(objs)(lambda o: p._contains_color(o)(2))
                # print('red_objs: {}'.format(red_objs))
                good = p._eq(p._length(red_objs))(1)
                # print('good: {}'.format(good))
                return good
            return p._construct_mapping3(good)(i)
        elif task == 55:
            return p._construct_mapping2('color')(i)
        elif task == 72:
            obj2 = lambda g: p._columns(g)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 80:
            obj2 = lambda g: p._objects2(g)(False)(False)
            objs = p._construct_mapping(obj2)(obj2)('rotation')(i)
            return p._place_into_grid(objs)
        elif task == 81:
            obj2 = lambda g: p._objects2(g)(True)(True)
            objs = p._construct_mapping(obj2)(obj2)('color')(i)
            return p._place_into_grid(objs)
        elif task == 96:
            objects = p._objects2(p._input(i))(True)(False)
            objects = p._filter_list(objects)(lambda o: p._not_pixel(o))
            return p._place_into_grid(objects)
        elif task == 102:
            return p._construct_mapping3(lambda g: p._has_y_symmetry(g))(i)
        elif task == 103:
            return p._construct_mapping2('rotation')(i)
        elif task == 185:
            return p._construct_mapping3(lambda g: p._area(g))(i)
        elif task == 229:
            obj2 = lambda g: p._objects2(g)(True)(False)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 234:
            obj2 = lambda g: p._objects2(g)(True)(True)
            return p._vstack(p._construct_mapping(obj2)(lambda g: p._rows(g))('none')(i))
        elif task == 261:
            obj2 = lambda g: p._rows(g)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 330:
            obj2 = lambda g: p._objects2(g)(False)(False)
            objs = p._construct_mapping(obj2)(obj2)('none')(i)
            return p._place_into_grid(objs)
        elif task == 398:
            return p._construct_mapping3(lambda g: p._area(g))(i)

    return prog


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
            # pd['size'],
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
        # featureExtractor=ArcNet,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python'
        # CPUs=5
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

    print('should have solved 14/14 tasks')



def run():
    test_construct_mapping()
    # for i in range(400):
        # check_solves(get_arc_task(i), program(i))

