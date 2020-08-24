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
        return p._rotate_ccw(p._rotate_ccw(p._input(i)))

    check_solves(task, program)


def task86():
    task = get_arc_task(86)

    def program(i):
        return p._rotate_ccw(p._rotate_ccw(p._input(i)))

    check_solves(task, program)

def task379():
    task = get_arc_task(379)

    def program(i):
        return p._rotate_ccw(p._input(i))

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
        return p._top_half(p._left_half(p._object(p._input(i))))

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


def task173():
    task = get_arc_task(173)

    def program(i):
        return p._get(p._filter_list(p._objects_by_color(p._input(i)))(lambda o: p._has_y_symmetry(o)))(0)

    check_solves(task, program)


def task359():
    task = get_arc_task(359)

    def program(i):
        return False

    check_solves(task, program)


def task243():
    task = get_arc_task(243)

    def program(i):
        return False

    check_solves(task, program)


def task210():
    task = get_arc_task(210)

    def program(i):
        return False

    check_solves(task, program)


def task350():
    task = get_arc_task(350)

    def program(i):
        return False

    check_solves(task, program)


def task241():
    task = get_arc_task(241)

    def program(i):
        return False

    check_solves(task, program)


def task360():
    task = get_arc_task(360)

    def program(i):
        return False

    check_solves(task, program)


def task141():
    task = get_arc_task(141)

    def program(i):
        return False

    check_solves(task, program)


def task151():
    task = get_arc_task(151)

    def program(i):
        return False

    check_solves(task, program)


def task105():
    task = get_arc_task(105)

    def program(i):
        return False

    check_solves(task, program)


def task240():
    task = get_arc_task(240)

    def program(i):
        return False

    check_solves(task, program)


def task19():
    task = get_arc_task(19)

    def program(i):
        return False

    check_solves(task, program)


def task73():
    task = get_arc_task(73)

    def program(i):
        return False

    check_solves(task, program)


def task82():
    task = get_arc_task(82)

    def program(i):
        return False

    check_solves(task, program)


def task26():
    task = get_arc_task(26)

    def program(i):
        return False

    check_solves(task, program)


def task70():
    task = get_arc_task(70)

    def program(i):
        return False

    check_solves(task, program)


def task102():
    task = get_arc_task(102)

    def program(i):
        return False

    check_solves(task, program)


def task180():
    task = get_arc_task(180)

    def program(i):
        return False

    check_solves(task, program)


def task61():
    task = get_arc_task(61)

    def program(i):
        return False

    check_solves(task, program)


def task108():
    task = get_arc_task(108)

    def program(i):
        return False

    check_solves(task, program)


def task111():
    task = get_arc_task(111)

    def program(i):
        return False

    check_solves(task, program)


def task116():
    task = get_arc_task(116)

    def program(i):
        return False

    check_solves(task, program)


def task247():
    task = get_arc_task(247)

    def program(i):
        return False

    check_solves(task, program)


def task375():
    task = get_arc_task(375)

    def program(i):
        return False

    check_solves(task, program)

def task52():
    task = get_arc_task(52)

    def program(i):
        # height six
        a = p._combine_grids_vertically(p._input(i))(p._x_mirror(p._input(i)))
        # bottom row is always black.
        b = p._top_half(p._x_mirror(p._input(i))) 
        # height seven
        c = p._combine_grids_vertically(b)(a) 
        # blank row stack over top two rows of the thing.
        d = p._top_half(c) 
        return d
        # return p._top_half(p._combine_grids_vertically(p._top_half(p._x_mirror(p._input(i))))(p._combine_grids_vertically(p._input(i))(p._x_mirror(p._input(i)))))

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
    task115()
    task173()
    # task359()
    # task243()
    # task210()
    # task350()
    # task241()
    # task360()
    # task141()
    # task151()
    # task105()
    # task240()
    # task19()
    # task73()
    # task82()
    # task26()
    # task70()
    # task102()
    # task180()
    # task61()
    # task108()
    # task111()
    # task116()
    # task247()
    # task375()
    task52()
