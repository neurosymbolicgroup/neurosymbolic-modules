import random
import numpy as np
from dreamcoder.domains.arc.arcPrimitives import ArcExample
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_map_task2(num_colors=7):
    def sample(i, j, k):
        return np.array([[i,j,k],[i,j,k],[i,j,k]])

    def sample_easy(i, j, k):
        return np.array([i,j,k])

    inp = 0
    def next_input():
        nonlocal inp
        inp += 1
        if inp == num_colors + 1:
            inp = 1
        return inp

    num_examples = math.ceil(num_colors/3)
    examples = []
    for i in range(num_examples):
        a, b, c = next_input(), next_input(), next_input()
        input_grid = ArcExample(sample_easy(a, b, c))
        output_grid = input_grid.transform(d).grid
        input_grid = input_grid.grid
        examples.append((input_grid, output_grid))

    return examples

def random_grid(easy=True):
    grid = ArcExample(np.random.randint(low=0,high=10,size=(1,3)))
    if not easy:
        grid = ArcExample(np.repeat(np.random.randint(
                        low=0, high=10, size=(1,3)), 3, axis=0))
    return grid

def random_grid2():
    # 1D grid, with numbers wrapped.
    i = list(range(1, 10))

def random_grid_and_program():
    num_mappings = random.randint(1, 9)
    inp = list(range(1, num_mappings+1))
    out = list(range(1, num_mappings+1))
    random.shuffle(inp)
    random.shuffle(out)
    transformation = zip(inp, out)
    program = ''
    for i, o in transformation:
        program += 'm' + str(i) + str(o)

    input_grid = np.array(inp)
    output_grid = np.array(out)
    ran = run_program(input_grid, program)
    assert np.array_equal(output_grid, ran)
    return input_grid, program

def run_program(i, p):
    o = np.copy(i)
    for m, k, v in np.array(list(p)).reshape(-1, 3):
        o[i == int(k)] = int(v)

    return o

def run():
    for i in range(10):
        print(random_grid_and_program())

def get_training_examples(batch_size=500):
    return [random_grid_and_program() for i in range(batch_size)]


def one_hot_grid(input_grid):
    # from
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    b = np.zeros((input_grid.size, input_grid.max()))
    b[np.arange(input_grid.size), input_grid-1] = 1
    return b

def undo_one_hot_grid(input_grid):
    return np.argmax(input_grid, axis=1)+1

class Net(nn.Module):
    def __init__(self):
        input_size = 9
        hidden_dim = 256
        output_size = 10
        self.lstm = nn.LSTM(input_size, 
                hidden_dim,
                batch_first=False,
                bidirectional=False)

        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
        x = one_hot_grid(x)
        x, _ = self.lstm(x, hidden)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
