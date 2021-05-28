import json
from functools import lru_cache
from typing import Dict, Tuple, Any, NamedTuple, List, Sequence
from matplotlib import colors
from matplotlib import pyplot as plt
from bidir.utils import assertEqual

import numpy as np

from bidir.primitives.types import Grid


class Task(NamedTuple):
    # list of values, each of which is tuple of examples
    inputs: Tuple[Tuple[Any, ...], ...]
    # list of examples
    target: Tuple[Any, ...]


def binary_task(target: int, start=2) -> Task:
    assertEqual(type(target), int, f"{target}")
    assertEqual(type(start), int, f"{start}")
    return Task(((start, ), ), (target, ))


def twenty_four_task(inputs: Tuple[int, ...], target: int) -> Task:
    return Task(tuple((i, ) for i in inputs), (target, ))


@lru_cache(maxsize=None)
def arc_task(task_num: int, train: bool = True) -> Task:
    train_exs, test_exs = get_arc_task_examples(task_num, train)
    input_exs, output_exs = zip(*train_exs)
    return Task((input_exs, ), output_exs)


@lru_cache(maxsize=None)
def get_arc_task_examples(
    task_num: int,
    train: bool = True,
) -> Tuple[Tuple[Tuple[Grid, Grid], ...], Tuple[Tuple[Grid, Grid], ...]]:
    """
    Returns a tuple (training_examples, test_examples), each of which is a
    tuple of examples, each example of which is a (Grid , Grid) tuple.

    train=False gives eval pairs
    """

    if train:
        task_path = "data/ARC/data/training/"
    else:
        task_path = "data/ARC/data/evaluation/"

    task_id = num_to_id(task_num, train=train)
    task_dict = load_task(task_id, task_path)

    train_examples = tuple(
        (Grid(x["input"]), Grid(x["output"])) for x in task_dict["train"])
    test_examples = tuple(
        (Grid(x["input"]), Grid(x["output"])) for x in task_dict["test"])

    return train_examples, test_examples


@lru_cache(maxsize=None)
def get_arc_grids() -> List[Grid]:
    """
    List of all grids used as inputs for a training or eval task.
    """
    grids = []
    for i in range(400):
        train_exs, test_exs = get_arc_task_examples(i, train=True)
        eval_train_exs, eval_test_exs = get_arc_task_examples(i, train=False)
        for (inp,
             outp) in train_exs + test_exs + eval_train_exs + eval_test_exs:
            grids.append(inp)
            # grids.append(outp)

    return grids


def num_to_id(task_num: int, train: bool = True) -> str:
    if train:
        id_path = 'data/ARC/task_number_ids.txt'
    else:
        id_path = 'data/ARC/eval_task_number_ids.txt'

    with open(id_path, 'r') as f:
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

    return task_dict


def plot_one(ax, grid, i, text):
    cmap = colors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA',
        '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = colors.Normalize(vmin=0, vmax=9)

    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(grid))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(grid[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(text)


def plot_grids(grids: Sequence[Sequence[np.ndarray]], text: Sequence[Sequence[str]]):
    num_test = len(grids)
    num_grids = max(len(g) for g in grids)
    fig, axs = plt.subplots(num_test,
                            num_grids,
                            figsize=(3 * num_grids, 3 * 1),
                            squeeze=False)
    for j in range(num_test):
        for i in range(len(grids[j])):
            plot_one(axs[j][i], grids[j][i], i, text[j][i])
    plt.tight_layout()


def int_to_color(i):
    d = {
        0: (0, 0, 0),
        1: (0, 118, 211),
        2: (255, 62, 61),
        3: (0, 204, 87),
        4: (255, 219, 72),
        5: (170, 170, 170),
        6: (247, 18, 184),
        7: (255, 132, 54),
        8: (120, 220, 252),
        9: (139, 8, 38),
    }

    return d[i]


def prepare_for_plot(task: Task) -> Sequence[Sequence[np.ndarray]]:
    assert len(task.inputs) == 1
    input_examples: Tuple[Grid, ...] = task.inputs[0]
    output_examples: Tuple[Grid, ...] = task.target

    input_examples2 = [g.arr for g in input_examples]
    output_examples2 = [g.arr for g in output_examples]

    all_grids = [input_examples2, output_examples2]
    return all_grids

def simple_text(grids):
    num_examples = len(grids[0])
    text = [['input'] * num_examples, ['output'] * num_examples]
    return text


def plot_task(task: Task, text: str = 'Hello', block=True):
    grids = prepare_for_plot(task)
    grid_text = simple_text(grids)
    plot_grids(grids, grid_text)
    plt.suptitle(text)
    plt.show(block=block)

    # plt.savefig('test.png')
    # print('Saved {}'.format(name))
    # plt.clf()
    # s = input()
    # if s == 'quit()': quit()
