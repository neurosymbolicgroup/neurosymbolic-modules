from dreamcoder.domains.arc.arcPrimitives import primitives, ArcExample
from dreamcoder.domains.arc.makeArcTasks import load_task, make_features
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives, ArcExample, t_arclist
from dreamcoder.program import Primitive
from dreamcoder.domains.arc.main import Sampler
import numpy as np
import random


def test_recognition(ec_result):
    def new_transform(p, i1, i2, i3):
        import copy
        p = copy.deepcopy(p)
        p.body.x = i1
        p.body.f.x = i2
        p.body.f.f.x = i3
        return p

    model = ec_result.recognitionModel
    task, solution = list(ec_result.taskSolutions.items())[0]
    program = solution.entries[0].program
    print('program: {}'.format(program))
    print('program body: {}'.format(program.body))
    print(type(program.body))
    print('program body: {}'.format(program.body.f))
    print(type(program.body.f))
    print('program body: {}'.format(program.body.f.f))
    print(type(program.body.f.f))
    # print('program body: {}'.format(program.body.f.f.f))
    # print(type(program.body.f.f.f))
    # print('program body: {}'.format(program.body.f.f.f.f))
    # print(type(program.body.f.f.f.f))
    # print('program body: {}'.format(program.body.f.f.f.f.f))
    # print(type(program.body.f.f.f.f.f))
    input = task.examples[0][0][0]
    output = program.evaluate([])(input)
    print('(input, output): {}'.format((input, output)))

    grammar = model.grammarOfTask(task)
    grammar = grammar.untorch()
    score = grammar.logLikelihood(task.request, program)

    n = primitives[-10:]

    task_id= "d07ae81c"
    d = load_task(task_id)
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["output"]))
            for training_example in d["train"]]


    for i, ex in enumerate(examples):
        results = []
        for i1 in range(10):
            for i2 in range(10):
                for i3 in range(10):
                    trio = i1, i2, i3
                    new_program = new_transform(program, n[i1], n[i2], n[i3])
                    output2 = new_program.evaluate([])(input)
                    score2 = grammar.logLikelihood(task.request, new_program)
                    results.append((trio, score2, new_program))


        results = sorted(results, key=lambda i: -i[1])
        with open('results' + str(i) + '.out', 'w') as f:
            for r in results:
                f.write(str(r) + '\n')


def make_test_ring_task():
    task_id = "85c4e7cd"
    d = load_task(task_id)
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["output"]))
            for training_example in d["train"]]

    # need examples to all be same size
    i, o = examples[0][0][0], examples[0][1]
    ex = i
    sampler = Sampler(i)
    examples_in = [ex] + [sampler.sample() for _ in range(5)]
    sol_fn = lambda i: _get_objects(i).apply_fn(lambda o: o.map_i_to_j(o.color(),
        _get_objects(i).reverse().get(_ix(o)).color())).stack()
    examples_out = list(map(sol_fn, examples_in))
    examples = [((ex_i,), ex_o) for (ex_i, ex_o) in zip(examples_in,
            examples_out)]

    print('i,: {}'.format(i,))
    print(str(i.get_objects()))
    print('o: {}'.format(o))
    print('examples: {}'.format(examples))
    expected = sol_fn(i)
    assert o == expected, "not good: {}, {}".format(o, expected)
    task = Task(task_id, 
            arrow(tgrid, tgrid),
            examples,
            make_features(examples))
    return task


# testing primtivies for the ring task
def test_ring_task():
    primitives = [
        Primitive("reverse_list", arrow(t_arclist, t_arclist), _reverse_list),
        Primitive("get", arrow(t_arclist, tint, tgrid), _get),
        Primitive("color", arrow(tgrid, tint), _color),
        Primitive("apply_fn", arrow(t_arclist, arrow(tgrid, tgrid), t_arclist), _apply_fn),
        Primitive("ix", arrow(tgrid, tint), _ix),
        Primitive("stack", arrow(t_arclist, tgrid), _stack),
        Primitive("get_objects", arrow(tgrid, t_arclist), _get_objects),
        # Primitive("transform", arrow(tgrid, tint, tint, tint, tint, tgrid), _transform),
        Primitive("mapitoj", arrow(tint, tint, tgrid, tgrid), _map_i_to_j_python),

    ]  + [Primitive(str(i), tint, i) for i in range(0, MAX_COLOR + 1)]

    task = make_test_ring_task()

    return primitives, [task], []



