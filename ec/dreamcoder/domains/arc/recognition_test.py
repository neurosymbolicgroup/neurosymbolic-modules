import numpy as np
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.domains.arc.arcPrimitives import Grid, Input, tgrid, tinput
from dreamcoder.domains.arc.arcPrimitives import _map_i_to_j, _get, _list_of, _pixels, _objects, _stack_no_crop
from dreamcoder.domains.arc.arcInput import export_tasks


def task1(seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, high=9, size=(20, 20))
    i1, o1, i2, o2 = rng.integers(0, high=5, size=4)
    # print('{} -> {}, {} -> {}'.format(i1, o1, i2, o2))
    outp = _map_i_to_j(Grid(inp))(i1)(o1)
    outp = _map_i_to_j(outp)(i2)(o2)
    outp = outp.grid

    examples = [{'input': inp, 'output': outp}]
    examples = [((Input(ex["input"], examples),), 
        Grid(ex["output"])) for ex in examples]

    task = Task('task1_' + str(seed), 
            arrow(tinput, tgrid),
            examples)
    return task

def task2(seed=0):
    obj = np.array([[1,1,1],[1,1,1],[1,1,1]])
    num_objects = 10
    grid = np.zeros((20, 20))
    rng = np.random.default_rng(seed)

    for i in range(num_objects):
        x, y = rng.integers(0, high=grid.shape[0]-3, size=2)
        grid[x:x+obj.shape[0], y:y+obj.shape[1]] += obj

    inp = grid
    succeeded = False
    attempts = 0
    while not succeeded:
        try:
            attempts += 1
            # high controls how hard enumeration is.
            o1, i1, o2, i2 = rng.integers(0, high=5, size=4)
            # print('o1, i1, o2, i2: {}'.format((o1, i1, o2, i2)))
            obj1 = _get(_objects(Grid(inp)))(o1)
            # obj2 = _get(_objects(Grid(inp)))(o2)
            pix1 = _get(_pixels(obj1))(i1)
            # pix2 = _get(_pixels(obj2))(i2)
            # outp = _stack_no_crop(_list_of(obj1)(obj2))
            outp = _absolute_grid(pix1)
            outp = outp.grid
            succeeded = True
        except ValueError:
            # print('attempts: {}'.format(attempts))
            succeeded = False
    
    examples = [{'input': inp, 'output': outp}]
    examples = [((Input(ex["input"], examples),), 
        Grid(ex["output"])) for ex in examples]

    task = Task('task2_' + str(seed), 
            arrow(tinput, tgrid),
            examples)
    return task


def run():
    tasks = [task1(), task2()]
    export_tasks('/home/salford/to_copy/', tasks)


def bin_file():
    # for task 1
    # primitives = [p['map_i_to_j'], p['input']]
    # colors = [p['color' + str(i)] for i in range(10)]
    # primitives = primitives + colors

    # for task 2
    primitives = ['get', 'objects', 'input', 'absolute_grid', 'pixels']
    primitives = [p[i] for i in primitives]
    ints = [p[str(i)] for i in range(5)]
    primitives = primitives + ints

    # combine tasks, hopefully won't solve
    primitives = ['get', 'objects', 'input', 'absolute_grid', 'pixels',
    'map_i_to_j']
    primitives = [p[i] for i in primitives]
    ints = [p[str(i)] for i in range(5)]
    colors = [p['color' + str(i)] for i in range(10)]
    primitives = primitives + ints + colors

    # create grammar
    grammar = Grammar.uniform(primitives)


    # generic command line options
    args = commandlineArguments(
        enumerationTimeout=7, 
        # activation='tanh',
        aic=.1, # LOWER THAN USUAL, to incentivize making primitives
        iterations=2, 
        # recognitionTimeout=60, 
        featureExtractor=ArcNet2,
        a=3, 
        maximumFrontier=10, 
        topK=1, 
        pseudoCounts=30.0,
        # helmholtzRatio=0.5, 
        # structurePenalty=.1, # HIGHER THAN USUAL, to incentivize making primitives
        solver='python',
        CPUs=5
        )

    # training = [task1(i) for i in range(10)]
    # training = [task2(i) for i in range(10)]
    training = [task1(i) for i in range(10)] + [task2(i) for i in range(10)]

    export_tasks('/home/salford/to_copy/', training)

    # iterate over wake and sleep cycles for our task
    generator = ecIterator(grammar,
                           training,
                           testingTasks=[],
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))
