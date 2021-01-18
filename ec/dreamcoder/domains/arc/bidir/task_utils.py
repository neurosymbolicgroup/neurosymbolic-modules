from typing import Tuple

from dreamcoder.domains.arc.bidir.primitives.types import Grid
from dreamcoder.domains.arc.arcInput import num_to_id, load_task


def get_task_grid_pairs(task_num: int, train: bool = True) -> Tuple[Grid, ...]:
    """train=False gives eval pairs"""
    if not train:
        raise NotImplementedError

    task_id = num_to_id(task_num)
    task_dict = load_task(task_id, task_path="data/ARC/data/training/")

    grid_pairs = tuple((Grid(x["input"]), Grid(x["output"]))
                       for x in task_dict["train"] + task_dict["test"])

    return grid_pairs
