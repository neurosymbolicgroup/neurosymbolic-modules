from dreamcoder.domains.arc.arcPrimitives import *
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcInput import load_task

# testing solving the tasks (don't want to import all of the primitives into
# other file
def make_arc_task(task_id):
    d = load_task(task_id)
    
    examples = [((Input(ex["input"], d["train"]),), 
        Grid(ex["output"])) for ex in d["train"]]
    examples += [((Input(ex["input"], d["train"]),),
        Grid(ex["output"])) for ex in d["test"]]

    task = Task(task_id, 
            arrow(tinput, tgrid),
            examples)
    return task

def check_solves(task, program):
    for i, ex in enumerate(task.examples):
        inp, out = ex[0][0], ex[1]
        predicted = program(inp)
        if predicted != out:
            print('inp: {}'.format(p._input(inp)))
            print('out: {}'.format(out))
            print('Failed example ' + str(i) + ': input=')
            print(p._input(inp))
            print('output=')
            print(out)
            print('predicted=')
            print(predicted)
            assert False, 'did NOT pass!'
            return
    print('Passed!')


def task1():
    task_id = 'f8b3ba0a'

    task = make_arc_task(task_id)

    # colors().sort(lambda c: grid.filter_col(c).objects().length()).vstack()
    def program(i):
        colors = p._colors(p._input(i))
        key_fn = lambda color: p._length(p._objects(p._filter_color(p._input(i))(color)))
        colors = p._remove_head(p._reverse(p._sort(colors)(key_fn)))
        colors = p._map(colors)(lambda color: p._pixel2(color))
        return p._vstack(colors)

    def program2(i):
        return p._vstack(p._map(p._remove_head(p._reverse(p._sort(p._colors(p._input(i)))(lambda color: p._length(p._objects(p._filter_color(p._input(i))(color)))))))(lambda color: p._pixel2(color)))

    check_solves(task, program2)

def task2():
    task_id = '0d3d703e' # map task

    task = make_arc_task(task_id)

    # objects().find_corresponding().stack()
    def program(i):
        objects = p._objects(p._input(i))
        # print('objects: {}'.format(objects))
        out_objs = p._map(objects)(lambda o: p._find_corresponding(i)(o))
        # print('out_objs: {}'.format(out_objs))
        out_grid = p._stack(out_objs)
        # print('out_grid: {}'.format(out_grid))
        return out_grid

    check_solves(task, program)

def task3():
    task_id = '995c5fa3' # map task 2.0

    task = make_arc_task(task_id)

    # objects().find_corresponding().stack()
    def program(i):
        objects = p._objects(p._input(i))
        # print('objects: {}'.format(objects))
        out_objs = p._map(objects)(lambda o: p._find_corresponding(i)(o))
        # print('out_objs: {}'.format(out_objs))
        out_grid = p._vstack(out_objs)
        # print('out_grid: {}'.format(out_grid))
        return out_grid

    check_solves(task, program)

def task4():
    task_id = '85c4e7cd' # concentric rings task

    task = make_arc_task(task_id)

    # objects().apply_colors(colors(objects().reverse())).stack()
    def program(i):
        objects = p._objects(p._input(i))
        # print('objects: {}'.format(objects))
        colors = p._map(p._reverse(p._objects(p._input(i))))(lambda o: p._color(o))
        # print('colors: {}'.format(colors))
        out_objs = p._apply_colors(objects)(colors)
        # print('out_objs: {}'.format(out_objs))
        out_grid = p._stack(out_objs)
        # print('out_grid: {}'.format(out_grid))
        return out_grid

    check_solves(task, program)

def task5():
    task_id = 'd511f180' # another map-esque task

    task = make_arc_task(task_id)

    # objects().find_corresponding().stack()
    # for each object in the input, convert it into something
    # "something": look in the i/o examples, and see where you've seen that
    # input before, and check the corresponding output object.

    # (lambda (stack ( map ( objects ( input $0)) lambda (find_corresponding $1 $0

    def program(i):
        objects = p._objects(p._input(i))
        # print('objects: {}'.format(objects))
        out_objs = p._map(objects)(lambda o: p._find_corresponding(i)(o))
        # print('out_objs: {}'.format(out_objs))
        out_grid = p._stack(out_objs)
        # print('out_grid: {}'.format(out_grid))
        return out_grid

    check_solves(task, program)

def task6():
    task_id = '2281f1f4' # weird pixel intersection task

    task = make_arc_task(task_id)

    # in.overlay(pixels().filter(lambda p: pixel(p.x, 0) == grey and pixel(10, p.y) == grey).stack().color_in(red))

    def program(i):
        pixels = p._pixels(p._input(i))
        print('pixels: {}'.format(pixels))
        grey = p._color(p._pixel(p._input(i))(0)(0))
        print('grey: {}'.format(grey))
        red_pixels = p._filter(pixels)(lambda p:
                p._and(p._eq(p._color(p._pixel(p._input(i))(p._x(p))(0)))(5))(
                    p._eq(p._color(p._pixel(p._input(i))(10)(p._y(p))))(5)))
        print('red_pixels: {}'.format(red_pixels))
        red_pixels = p._map(red_pixels)(lambda p: p._color(p)(red))
        print('red_pixels: {}'.format(red_pixels))
        red_pixels = p._stack(red_pixels)
        output = p._overlay(red_pixels)(p._input(i))
        print('output: {}'.format(output))
        return out_grid

    check_solves(task, program)

def task7():
    task_id = 'be94b721'

    task = make_arc_task(task_id)

    def program(i):
        return p._get(p._reverse(p._sort(p._objects(p._input(i)))(lambda o: p._area(o))))(0)

    check_solves(task, program)

def task8():
    #TODO: needs "contiguous objects" primitive in order to work
    task_id = '42a50994'

    def program(i):
        blues = p._set_shape(p._stack(p._filter(p._objects(p._input(i)))(lambda o:
            p._not(p._eq(p._area(o))(1)))))(p._shape(p._input(i)))
        return blues



    task = make_arc_task(task_id)
    check_solves(task, program)

def task9():
    task_id = '4347f46a' #hollow
    task = make_arc_task(task_id)
    
    def program(i):
        return p._hollow_objects(p._input(i))
    
    check_solves(task, program)


def task10():
    task_id = 'a699fb00' #fill line
    task = make_arc_task(task_id)

    def program(i):
        return p._fill_line(p._input(i))
    
    check_solves(task, program)


def task11():
    task_id = '496994bd' #horizontal plane mirroring
    task = make_arc_task(task_id)

    def program(i):
        return p._horizontal_mirroring(p._input(i))
    
    check_solves(task, program)

def task12():
    task_id = '7fe24cdd' #rotating grid and adding each rotation to different parts of output grid
    task = make_arc_task(task_id)
    def program(i):
        return p._combine_grids_vertically(p._combine_grids_horizontally(p._input(i), p._clockwise_rotate(p._input(i))),
                    p._combine_grids_horizontally(p._clockwise_rotate(p._clockwise_rotate(p._clockwise_rotate(p._input(i)))),
                    p._clockwise_rotate(p._clockwise_rotate(p._input(i)))))
    check_solves(task, program)


def run():
    #task1()
    #task2()
    #task3()
    #task4()
    # task5()
    # task6()
    #task7()
    # task8()
    #task9()
    #task10()
    #task11()
    task12()

def full_arc_task(include_eval=False):
    training_dir = 'data/ARC/data/training/'
    evaluation_dir = 'data/ARC/data/evaludation/'

    # take off last five chars of name to get rid of '.json'
    task_ids = [t[:-5] for t in os.listdir(training_dir)]
    if include_eval:
        task_ids += [t[:-5] for t in os.listdir(evaluation_dir)]

    return [make_arc_task(task_id) for task_id in task_ids]

def get_tasks():
    return [make_arc_task('f8b3ba0a')], []
    # return full_arc_task(), []
