from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.makeTasks import make_arc_task
import dreamcoder.domains.arc.arcPrimitives as p
from dreamcoder.domains.arc.arcInput import load_task

def check_solves(task, program):
    for i, ex in enumerate(task.examples):
        inp, out = ex[0][0], ex[1]
        predicted = program(inp)
        if predicted != out:
            # print('inp: {}'.format(p._input(inp)))
            # print('out: {}'.format(out))
            # print('Failed example ' + str(i) + ': input=')
            # print(p._input(inp))
            # print('output=')
            # print(out)
            # print('predicted=')
            # print(predicted)
            # assert False, 'did NOT pass!'
            print('Did not pass')
            return
    print('Passed!')


def task379():
    task_id = 'ed36ccf7'

    task = make_arc_task(task_id)

    def program(i):
        return p._clockwise_rotate(.p._input(i))

    check_solves(task, program)


def task139():
    task_id = '6150a2bd'

    task = make_arc_task(task_id)

    def program(i):
        return p._clockwise_rotate(p._clockwise_rotate(p._input(i)))

    check_solves(task, program)


def task86():
    task_id = '3c9b0459'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task149():
    task_id = '67a3c6ac'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task154():
    task_id = '68b16354'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task209():
    task_id = '8be77c9e'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task171():
    task_id = '6fa7a44f'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task163():
    task_id = '6d0aefbc'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task38():
    task_id = '2013d3e2'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task112():
    task_id = '496994bd'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task115():
    task_id = '4c4377d9'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task173():
    task_id = '72ca375d'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task359():
    task_id = 'e3497940'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task243():
    task_id = '9f236235'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task210():
    task_id = '8d5021e8'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task350():
    task_id = 'dc0a314f'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task241():
    task_id = '9ecd008a'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task360():
    task_id = 'e40b9e2f'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task141():
    task_id = '62c24649'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task151():
    task_id = '67e8384a'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task105():
    task_id = '46442a0e'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task240():
    task_id = '9dfd6313'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task19():
    task_id = '11852cab'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task73():
    task_id = '3631a71a'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task82():
    task_id = '3af2c5a8'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task26():
    task_id = '1b60fb0c'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task70():
    task_id = '3345333e'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task102():
    task_id = '44f52bb0'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task180():
    task_id = '760b3cac'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task61():
    task_id = '2bcee788'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task108():
    task_id = '47c1f68c'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task111():
    task_id = '4938f0c2'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task116():
    task_id = '4c5c2cf0'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task247():
    task_id = 'a3df8b1e'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def task375():
    task_id = 'eb281b96'

    task = make_arc_task(task_id)

    def program(i):
        return False

    check_solves(task, program)


def run():
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    task8()
    task9()
    task10()
    task11()
    task12()
