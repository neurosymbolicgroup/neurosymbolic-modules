from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist
from dreamcoder.domains.arc.arcPrimitives import tgrid, primitives, ArcExample, _gridempty
from dreamcoder.domains.arc.arcInput import load_task

def make_task(task_id):
    d = load_task(task_name)
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["output"]))
            for training_example in d["train"]]
    print('examples: {}'.format(examples))
    task = Task(task_d, 
            arrow(tgrid, tgrid),
            examples,
            make_features(examples))
    return task

def make_tasks():
    task_name = "d07ae81c"
    d = load_task(task_name)

    # ---------------------------------------------
    # TASK where input grid is same as output grid
    # ---------------------------------------------
    
    examples = [((ArcExample(training_example["input"]),), 
            ArcExample(training_example["input"]))
            for training_example in d["train"]]

    task_identity = Task(
            task_name + "IDENTITY",
            arrow(tgrid, tgrid),
            examples,
            make_features(examples)
        )

    examples = [((ArcExample(training_example["input"]),),
            _gridempty(ArcExample(training_example["input"])))
            for training_example in d["train"]]

    # ---------------------------------------------
    # TASK that takes in grid and outputs blank grid of same shape as INPUT
    # ---------------------------------------------
    
    task_blank_in = Task(task_name + "BLANK_IN",
            arrow(tgrid, tgrid),
            examples,
            make_features(examples)
        )

    # ---------------------------------------------
    # TASK that maps 3 colors
    # ---------------------------------------------
    
    array1_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array1_out = [[4, 5, 2], 
                  [4, 5, 2], 
                  [4, 5, 2]]
    arc1_in = ArcExample(array1_in)
    arc1_out = ArcExample(array1_out)
    should_be = arc1_in.map_i_to_j(3, 4).map_i_to_j(1, 5)#.map_i_to_j(2, 3)
    assert arc1_out == should_be, 'incorrect example created'

    example = (arc1_in,), arc1_out
    examples1 = [example]

    task_1 = Task(
            task_name + "2_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples1,
            features=make_features(examples)
        )

    # ---------------------------------------------
    # TASK that maps 2 colors
    # ---------------------------------------------
    
    array2_in = [[3, 1, 2], 
                 [3, 1, 2], 
                 [3, 1, 2]]
    array2_out = [[4, 5, 3], 
                  [4, 5, 3], 
                  [4, 5, 3]]
    arc2_in = ArcExample(array2_in)
    arc2_out = ArcExample(array2_out)
    should_be = arc2_in.map_i_to_j(3, 4).map_i_to_j(1, 5).map_i_to_j(2, 3)
    assert arc2_out == should_be, 'incorrect example created'

    example = (arc2_in,), arc2_out
    examples2 = [example]

    # ex: ((arc1_in,), arc1_out), tuple of length one?
    # ex[0]: 

    

    task_2 = Task(
            task_name + "3_MAP_COLORS",
            arrow(tgrid, tgrid),
            examples2,
            features=make_features(examples)
        )

    print(task_1.examples)
    print(task_2.examples)

    # training = [task_identity, task_blank_in, task_1]
    # testing = [task_identity]

    training = [task_1, task_2]
    testing = []

    return training, testing


def make_features(examples):
    # [ ( (arc1_in,), arc1_out) ]
    features = []
    for ex in examples:
        inp, outp = ex[0][0], ex[1]
        features.append(inp.input_list)
        features.append(outp.input_list)
    
    return features



