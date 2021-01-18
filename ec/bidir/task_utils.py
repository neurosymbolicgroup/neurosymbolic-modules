import json
from typing import Dict, Tuple

import numpy as np

from bidir.primitives.types import Grid


def get_task_grid_pairs(
    task_num: int,
    train: bool = True,
) -> Tuple[Tuple[Grid, Grid], ...]:
    """train=False gives eval pairs"""
    if not train:
        raise NotImplementedError

    task_id = num_to_id(task_num)
    task_dict = load_task(task_id, task_path="data/ARC/data/training/")

    grid_pairs = tuple((Grid(x["input"]), Grid(x["output"]))
                       for x in task_dict["train"] + task_dict["test"])

    return grid_pairs


def num_to_id(task_num: int) -> str:
    with open('dreamcoder/domains/arc/task_number_ids.txt', 'r') as f:
        lines = [line.rstrip() for line in f]
    d = {
        int(line.split(' ')[0]): line.split(' ')[-1].rstrip(".json")
        for line in lines
    }
    return d[task_num]


def load_task(
    task_id: str,
    task_path: str = 'data/ARC/data/training/',
) -> Dict:
    filename = task_path + task_id + '.json'

    with open(filename, 'r') as f:
        task_dict = json.load(f)

    task_dict['name'] = task_id

    # turn to np arrays
    train = task_dict["train"]
    for ex in train:
        for key in ex:
            ex[key] = np.array(ex[key])

    test = task_dict["test"]
    for ex in test:
        for key in ex:
            ex[key] = np.array(ex[key])

    # print(task_dict)

    return task_dict
