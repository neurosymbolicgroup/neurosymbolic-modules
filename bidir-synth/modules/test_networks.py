from bidir.task_utils import get_arc_task_examples
from bidir.primitives.types import Grid
from rl.ops.arc_ops import OP_DICT
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time


def get_grids():
    grids = []
    for ex_i in range(400):
        train_exs, test_exs = get_arc_task_examples(ex_i)

        for (i, o) in list(train_exs) + list(test_exs):
            grids.append(i)
            grids.append(o)

    return grids


def get_operators():
    names = ['rotate_cw', 'rotate_ccw', 'hflip', 'vflip', 'top_half']
    return {name: OP_DICT[name] for name in names}


def export_data(samples, path):
    # samples is (grid, grid, name)
    def to_str(grid):
        return str(grid.shape) + ', ' + ' '.join(map(str, list(
            grid.flatten())))

    def line_of_example(example):
        if len(example) < 3:
            print(f"example: {example}")
        inp, outp, op = example
        return 'IN: {}, OUT: {}, OP: {}'.format(to_str(inp.arr),
                                                to_str(outp.arr), op)

    lines = ''.join(line_of_example(s) + '\n' for s in samples)
    with open(path, 'w+') as f:
        f.write(lines)


def generate_dataset():
    grids = get_grids()
    grids = [g for g in grids if min(g.arr.shape) > 2]
    print('{} grids'.format(len(grids)))

    operators = get_operators()
    print('{} operators'.format(len(operators)))

    num_unique = 0
    num_total = len(operators) * len(grids)
    print('num_total: {}'.format(num_total))
    samples = [(grid, op.fn.fn(grid), name) for grid in grids
               for name, op in operators.items()]
    # for grid in grids:
    #     outputs = []
    #     for op in operators.values():
    #         match = np.any([np.all(o == op.fn.fn(grid)) for o in outputs])
    #         if not match:
    #             num_unique += 1
    #             outputs.append(op.fn.fn(grid))

    # print('num_unique: {}'.format(num_unique))

    # export_data(samples, 'data/arcnet_data_new.txt')
    data = import_data('data/arcnet_data_new.txt')
    assert samples == data
    # export_data(data, 'data/arcnet_data_new2.txt')


def import_data(path):
    def parse_example(line):
        def parse_grid(grid_txt):
            i1 = grid_txt.index('(')
            assert i1 == 0
            i2 = grid_txt.index(')')
            shape = grid_txt[i1 + 1:i2]
            shape = shape.split(', ')
            w, h = list(map(int, shape))

            rest = grid_txt[grid_txt.index(')') + 3:]
            one_d = np.fromstring(rest, dtype=int, sep=' ')
            out = one_d.reshape(w, h)
            return out

        i1 = line.index('IN: ')
        i2 = line.index('OUT: ')
        i3 = line.index('OP: ')
        grid1 = line[i1 + 4:i2 - 2]
        grid2 = line[i2 + 5:i3 - 2]
        op = line[i3 + 4:-1]
        grid1 = parse_grid(grid1)
        grid2 = parse_grid(grid2)
        return Grid(grid1), Grid(grid2), op

    with open(path, 'r') as f:
        lines = f.readlines()
        return [parse_example(l) for l in lines]


def make_features(examples):
    # zero pad, concatenate, one-hot encode
    def pad(i):
        a = np.zeros((30, 30))
        if i.size == 0:
            return a

        # if input is larger than 30x30, crop it. Must be a created grid
        i = i[:min(30, len(i)), :min(30, len(i[0]))]
        a[:len(i), :len(i[0])] = i
        return a

    examples = [(pad(ex[0][0].input_grid.grid), pad(ex[1].grid))
                for ex in examples]
    examples = [
        torch.from_numpy(np.concatenate(ex)).to(torch.int64) for ex in examples
    ]
    input_tensor = F.one_hot(torch.stack(examples), num_classes=10)
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    # (num_examples, num_colors, h, w)
    return input_tensor


def train():
    torch.set_num_threads(10)
    # list of (grid1, grid2, op_str)
    data = import_data('arcnet_data_new.txt')

    ops = sorted(list(set([d[1] for d in data])))
    op_dict = dict(zip(ops, range(len(ops))))

    # convert (examples, op_str) to (input_tensor, target_tensor)
    def tensorize(datum):
        (examples, op_str) = datum
        # (1, num_colors, h, w)
        input_tensor = make_features(examples)
        # (1)
        target_tensor = torch.tensor(op_dict[op_str])
        return input_tensor, target_tensor

    data = [tensorize(d) for d in data]

    net = FullNet(len(op_dict))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 16
    epochs = 3100

    def make_batches(data, batch_size):
        return [
            data[batch_size * i:min((i + 1) * batch_size, len(data))]
            for i in range(math.ceil(len(data) / batch_size))
        ]

    print('Starting training...')
    for epoch in range(1, epochs + 1):
        random.shuffle(data)
        batches = make_batches(data, batch_size)
        start = time.time()

        total_loss = 0
        total_correct = 0
        for i, batch in enumerate(batches):
            # batch is list of (in_grid, out_grid, op_str) tuples
            examples, targets = zip(*batch)

            print(examples[0].shape)
            examples = torch.cat(examples)
            print('examples: {}'.format(examples))
            assert False

            targets_tensor = torch.tensor(targets)

            optimizer.zero_grad()

            out = net(examples)
            out = [t.unsqueeze(0) for t in out]
            out = torch.cat(out)
            predictions = torch.argmax(out, dim=1)
            # print('predictions: {}'.format(predictions))

            loss = criterion(out, targets_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.sum().item()

            num_correct = sum(t == p for t, p in zip(targets, predictions))
            # print('num_correct: {}'.format(num_correct))

            total_correct += num_correct

        accuracy = 100 * total_correct / len(data)
        duration = time.time() - start
        m = math.floor(duration / 60)
        s = duration - m * 60
        duration = f'{m}m {int(s)}s'

        print(
            f'Epoch {epoch} completed ({duration}) accuracy: {accuracy:.2f} loss: {loss:.2f}'
        )

    print('Finished Training')
