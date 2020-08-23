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

# def task374():
#     """
#     From a point, draw diagonal lines in all four directions
#     """

#     task = get_arc_task(140)

#     def program(i):
#         color = p._color(p._get(p._objects(p._input(i)))(0)) # get the first object color

#         start_pos=p._position(p._get(p._objects(p._input(i)))(0)) # get the first object position

#         line1 = p._draw_line(p._input(i),start_pos,d=45) # draw first line
#         line2 = p._draw_line(p._input(i),start_pos,d=-45) # draw second line

#         bothlines = p._overlay(line1)(line2) # stack the lines

#         return p._color_in_grid(bothlines)(color) # color them

#     check_solves(task, program)


def task140():
    """
    From a point, draw diagonal lines in all four directions, the same color as that point
    """
    task = get_arc_task(140)

    def program(i):
        color = p._color(p._get(p._objects(p._input(i)))(0)) # get the first object color

        start_pos=p._position(p._get(p._objects(p._input(i)))(0)) # get the first object position

        line1 = p._draw_line(p._input(i),start_pos,d=45) # draw first line
        line2 = p._draw_line(p._input(i),start_pos,d=-45) # draw second line

        bothlines = p._overlay(line1)(line2) # stack the lines

        return p._color_in_grid(bothlines)(color) # color them

    check_solves(task, program)



# def task355():
#     """
#     connect points of same color
#     """

#     # for each object: draw line until it hits other object...but not if it goes off grid
#     pass



def task36():
    """
    connect points of same color
    """

    task = get_arc_task(36)

    def program(i):

        # [o1, o2...o6]
        obs = p._objects(p._input(i))

        # [color of o1, color of o2... color of o6]
        ob_colors = p._map(obs)(p._color)

        # [func comparing color of o1...func comparing color of o6]
        funcs = p._map(obs)(p._compare(p._color))

        # [[objs that have color of o1], [obs that have color of o2]...]
        samecolorobjs = p._map (funcs) ( p._filter_list(obs) ) 

        # lines in correct locations
        bwlines = p._zip (obs) (samecolorobjs) (  p._draw_connecting_line(p._input(i))  )

        # lines with correct colors
        coloredlines = p._zip(bwlines) (ob_colors) (p._color_in_grid)

        # stack
        final = p._stack_no_crop(coloredlines)
        

        return final

    check_solves(task, program)

    



def run():
    task140()
    task36()
