from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.makeTasks import get_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcInput import load_task

def check_solves(task, program):
    for i, ex in enumerate(task.examples):
        inp, out = ex[0][0], ex[1]
        predicted = program(inp)
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


def task139():
    task = get_arc_task(139)

    def program(i):
        return p._clockwise_rotate(p._clockwise_rotate(p._input(i)))

    check_solves(task, program)


def task86():
    task = get_arc_task(86)

    def program(i):
        return p._clockwise_rotate(p._clockwise_rotate(p._input(i)))

    check_solves(task, program)

def task379():
    task = get_arc_task(379)

    def program(i):
        return p._clockwise_rotate(p._clockwise_rotate(p._clockwise_rotate(p._input(i))))

    check_solves(task, program)




def task149():
    task = get_arc_task(149)

    def program(i):
        return p._y_mirror(p._input(i))

    check_solves(task, program)


def task154():
    task = get_arc_task(154)

    def program(i):
        return p._x_mirror(p._input(i))

    check_solves(task, program)


def task209():
    task = get_arc_task(209)

    def program(i):
        return p._combine_grids_vertically(p._input(i))(p._x_mirror(p._input(i)))

    check_solves(task, program)


def task171():
    task = get_arc_task(171)

    def program(i):
        return p._combine_grids_vertically(p._input(i))(p._x_mirror(p._input(i)))

    check_solves(task, program)


def task163():
    task = get_arc_task(163)

    def program(i):
        return p._combine_grids_horizontally(p._input(i))(p._y_mirror(p._input(i)))

    check_solves(task, program)


def task38():
    task = get_arc_task(38)

    def program(i):
        return p._top_half(p._left_half(p._get_object(p._input(i))))

    check_solves(task, program)


def task112():
    task = get_arc_task(112)

    def program(i):
        return p._overlay(p._x_mirror(p._input(i)))(p._input(i))

    check_solves(task, program)


def task115():
    task = get_arc_task(115)

    def program(i):
        return p._combine_grids_vertically(p._x_mirror(p._input(i)))(p._input(i))

    check_solves(task, program)

def task140():
    task = get_arc_task(140)

    def program(i):
        return p._draw_line(p._input(i) )

    check_solves(task, program)


def run():
    task379()
    task139()
    task86()
    task149()
    task154()
    task209()
    task171()
    task163()
    task38()
    task112()
    task140()
