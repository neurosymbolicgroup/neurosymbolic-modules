import numpy as np
from numpy.random import default_rng
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.task import Task
from dreamcoder.domains.logo.main import Flatten
from dreamcoder.domains.arc.arcPrimitives import Grid, MAX_COLOR, Input
from dreamcoder.domains.arc.modules import *
from dreamcoder.domains.arc.recognition_test import task1, task2, task_size, shuffle_task

class ArcNet2(nn.Module):
    # for testing recognition network
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super().__init__()

        # maybe this should be false, but it doesn't hurt to make it true, so
        # doing that to be safe. See recognition.py line 908
        self.recomputeTasks = True

        self.num_examples_list = [len(t.examples) for t in tasks]

        intermediate_dim = 64 # output of all_conv
        lstm_hidden = 64
        # need to keep this named this for some other part of dreamcoder.
        self.outputDimensionality = lstm_hidden

        self.all_conv = AllConv(residual_blocks=1,
                residual_filters=10,
                conv_1x1s=1,
                output_dim=intermediate_dim,
                conv_1x1_filters=64,
                pooling='max')

        self.lstm = nn.LSTM(
            intermediate_dim,
            lstm_hidden, 
            batch_first=False,
            bidirectional=False)


    def forward(self, x):
        # (num_examples, num_colors, h, w) to (num_examples, intermediate_dim)
        x = x.to(torch.float32)
        x = self.all_conv(x)
        # print('x: {}'.format(x.shape))

        # (num_examples, intermediate_dim) to (num_examples, 1, intermediate_dim)
        # x = x.unsqueeze(1) 
        # only care about final hidden state
        # _, (x, _) = self.lstm(x)

        # (1, 1, outputDimensionality) to (outputDimensionality)
        # x = x.squeeze()
        return x

    def make_features(self, examples):
        # zero pad, concatenate, one-hot encode
        def pad(i):
            a = np.zeros((30, 30))
            a[:len(i), :len(i[0])] = i
            return a

        examples = [(pad(ex[0][0].input_grid.grid), pad(ex[1].grid))
                for ex in examples]
        examples = [torch.from_numpy(np.concatenate(ex)).to(torch.int64)
                for ex in examples]
        input_tensor = F.one_hot(torch.stack(examples), num_classes=10)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        # (num_examples, num_colors, h, w)
        return input_tensor

    #  we subclass nn.module, so this calls __call__, which calls forward()
    #  above
    def featuresOfTask(self, t): 
        # t.features is created when the task is made in makeArcTasks.py
        return self(self.make_features(t.examples))


    # make tasks out of the program, to learn how the program operates (dreaming)
    def taskOfProgram(self, p, t):
        num_examples = random.choice(self.num_examples_list)

        def generate_sample():
            inp, outp, lengths = shuffle_task(n=task_size())
            inp = Input(inp.grid, training_examples=[])

            try: 
                outp = p.evaluate([])(inp)
                example = (inp,), outp
                return example
            except ValueError:
                return None

        examples = [generate_sample() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples)
        return t


class ArcNet(nn.Module):
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super().__init__()

        # for sampling input grids during dreaming
        self.grids = []
        for t in tasks:
            self.grids += [ex[0][0] for ex in t.examples]

        # maybe this should be false, but it doesn't hurt to make it true, so
        # doing that to be safe. See recognition.py line 908
        self.recomputeTasks = True

        self.num_examples_list = [len(t.examples) for t in tasks]

        # need to keep this named this for some other part of dreamcoder.
        self.outputDimensionality = 64

        self.all_conv = AllConv(residual_blocks=1,
                residual_filters=10,
                conv_1x1s=1,
                output_dim=self.outputDimensionality,
                conv_1x1_filters=64,
                pooling='max')

    def forward(self, x):
        # (num_examples, num_colors, h, w) to (num_examples, intermediate_dim)
        x = x.to(torch.float32)
        x = self.all_conv(x)

        # sum features over examples
        # (num_examples, intermediate_dim) to (intermediate_dim)
        x = torch.sum(x, 0)

        return x

    def make_features(self, examples):
        # zero pad, concatenate, one-hot encode
        def pad(i):
            a = np.zeros((30, 30))
            if i.size == 0:
                return a

            # if input is larger than 30x30, crop it. Must be a created grid
            i = i[:min(30, len(i)),:min(30, len(i[0]))]
            a[:len(i), :len(i[0])] = i
            return a

        examples = [(pad(ex[0][0].input_grid.grid), pad(ex[1].grid))
                for ex in examples]
        examples = [torch.from_numpy(np.concatenate(ex)).to(torch.int64)
                for ex in examples]
        input_tensor = F.one_hot(torch.stack(examples), num_classes=10)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        # (num_examples, num_colors, h, w)
        return input_tensor

    #  we subclass nn.module, so this calls __call__, which calls forward()
    #  above
    def featuresOfTask(self, t): 
        # t.features is created when the task is made in makeArcTasks.py
        return self(self.make_features(t.examples))


    # make tasks out of the program, to learn how the program operates (dreaming)
    def taskOfProgram(self, p, t):
        num_examples = random.choice(self.num_examples_list)

        def generate_sample():
            grid = np.random.choice(self.grids)
            try: 
                out = p.evaluate([])(grid)
                example = (grid,), out
                return example
            except ValueError:
                return None

        examples = [generate_sample() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples)
        return t


class ArcFeatureNN(nn.Module):
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super(ArcFeatureNN, self).__init__()

        self.sampler = Sampler(tasks[0].examples[0][0][0])

        self.recomputeTasks = False

        self.num_examples_list = [len(t.examples) for t in tasks]

        # assume they're all the same size
        self.input_dim = len(tasks[0].features)
        # need to keep this named this for some other part of dreamcoder.
        self.outputDimensionality = 100
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.outputDimensionality),
            nn.ReLU(),
            nn.Linear(self.outputDimensionality, self.outputDimensionality),
            nn.ReLU(),
            nn.Linear(self.outputDimensionality, self.outputDimensionality),
            nn.ReLU()
        )

    def forward(self, v):
        v = torch.tensor(v).float()
        v = self.encoder(v)
        return v


    #  we subclass nn.module, so this calls __call__, which calls forward()
    #  above
    def featuresOfTask(self, t): 
        # t.features is created when the task is made in makeArcTasks.py
        return self(t.features)

    # make tasks out of the program, to learn how the program operates (dreaming)
    def taskOfProgram(self, p, t):
        num_examples = random.choice(self.num_examples_list)

        def generate_example():
            # makes the striped grid with random numbers
            grid = Grid(np.repeat(np.random.randint(
                    low=0, high=10, size=(1,3)), 3, axis=0))
            out = p.evaluate([])(grid)
            example = (grid,), out
            return example

        def generate_sample():
            grid = self.sampler.sample()
            try: 
                out = p.evaluate([])(grid)
                example = (grid,), out
                return example
            except ValueError:
                return None

        def generate_andy_example():
            grid = Grid(np.random.randint(low=0,high=10,size=(1,3)))
            try: 
                out = p.evaluate([])(grid)
                example = (grid,), out
                return example
            except ValueError:
                return None

        def generate_easy_example():
            n = num_pixels()
            rng = default_rng()
            example_n = rng.choice(range(1, n + 1))
            grid = np.zeros(n)
            grid[:example_n] = np.random.randint(low=0, high=10, size=example_n)
            grid = Grid(grid)

            out = p.evaluate([])(grid)
            example = (grid,), out
            return example

        examples = [generate_example() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples, features=make_features(examples))
        return t

class SamplerOld():
    def __init__(self, grid):
        self.grid = grid

    def sample(self):
        l = [random.randint(1, MAX_COLOR) for i in range(0, MAX_COLOR+1)]
        new_grid = self.grid.transform({i: l[i] for i in range(len(l))})
        return new_grid

