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

def task128():
    """
    Color the whole grid the color of most frequent color
    """
    task = get_arc_task(128)

    def program(i):
        g = p._input(i)
        # c = p._color(p._get_last(p._sort(
        #         p._map(p._colors(g))(p._filter_color(g)) # color separated grids
        #     )(p._area)))
        out = p._flood_fill(g)(p._color(g))
        return out

        # g = p._input(i)
        # c = p._color(p._get_last(p._sort(
        #         p._map(p._colors(g))(p._filter_color(g)) # color separated grids
        #     )(p._area)))
        # out = p._flood_fill(g)(c)
        # return out

    check_solves(task, program)



def task140():
    """
    From a point, draw diagonal lines in all four directions, the same color as that point
    """
    task = get_arc_task(140)

    def program(i):

        return p._color_in_grid(
            p._overlay(
                    p._draw_line(p._input(i))(p._get(p._objects(p._input(i)))(0))(45)
                )(
                    p._draw_line(p._input(i))(p._get(p._objects(p._input(i)))(0))(315)
                )
            )(p._color(p._get(p._objects(p._input(i)))(0)))


        # o = p._get(p._objects(p._input(i)))(0) # get first object

        # line1 = p._draw_line(p._input(i))(o)(45) # draw first line
        # line2 = p._draw_line(p._input(i))(o)(315) # draw second line

        # bothlines = p._overlay(line1)(line2) # stack the lines

        # return p._color_in_grid(bothlines)(p._color(o)) # color them

    check_solves(task, program)




def task36():
    """
    connect points of same color
    """

    task = get_arc_task(36)

    def program(i):

        # [o1, o2...o6]
        obs = p._objects(p._input(i))

        # [color of o1, color of o2... color of o6]
        # 2, 4, 4, 6, 2, 6
        ob_colors = p._map(p._color)(obs)

        # [func comparing color of o1...func comparing color of o6]
        funcs = p._map(p._compare(p._color))(obs)

        # [[objs that have color of o1], [obs that have color of o2]...]
        # [[obj 1 and ob 5], [obj 2 and ob 3]...]
        samecolorobjs = p._map  ( p._filter_list(obs) ) (funcs)

        # lines in correct locations
        bwlines = p._zip (obs) (samecolorobjs) (  p._draw_connecting_line(p._input(i))  )

        # lines with correct colors
        coloredlines = p._zip(bwlines) (ob_colors) (p._color_in_grid)

        # stack
        final = p._stack_no_crop(coloredlines)
        

        return final

    check_solves(task, program)

    



def run():
    task128()
    task140()
    task36()
