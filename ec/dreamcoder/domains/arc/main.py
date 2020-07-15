import numpy as np
from numpy.random import default_rng
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.task import Task
from dreamcoder.domains.logo.main import Flatten
from dreamcoder.domains.arc.arcPrimitives import ArcExample, MAX_COLOR
from dreamcoder.domains.arc.makeArcTasks import make_features
# easiest way to have a global variable
from dreamcoder.domains.arc.makeArcTasks import num_pixels

class ResBlock(nn.Module):
    def __init__(self, conv1_filters, conv2_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv2_filters, conv2_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_filters)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out

class OneByOneBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = nn.Conv1d(in_filters, out_filters, 1)
        self.bn1 = nn.BatchNorm1d(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


class Sampler():
    def __init__(self, grid):
        self.grid = grid

    def sample(self):
        l = [random.randint(1, MAX_COLOR) for i in range(0, MAX_COLOR+1)]
        new_grid = self.grid.transform({i: l[i] for i in range(len(l))})
        return new_grid

class Sampler2():
    def __init__(self, tasks):
        self.tasks = tasks
        # sample randomly from all of the input grids
        self.grids = []
        for t in tasks:
            self.grids += [ex[0][0] for ex in t.examples]
    
    def sample(self):
        return np.random.choice(self.grids)


class ArcNet(nn.Module):
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super().__init__()

        self.sampler = Sampler2(tasks[0].examples[0][0][0])

        # maybe this should be false, but it doesn't hurt to make it true, so
        # doing that to be safe. See recognition.py line 908
        self.recomputeTasks = True

        self.num_examples_list = [len(t.examples) for t in tasks]

        self.num_blocks = 1 # number of residual bocks (two 3x3 convs)
        self.num_filters = 10
        self.num_1x1_convs = 2 # number of 1x1 conv layers
        self.intermediate_dim = 256
        self.filters_1x1 = [128] * (self.num_1x1_convs) + [self.intermediate_dim]
        self.pooling = 'max' # max or average
        self.lstm_hidden = 128

        # need to keep this named this for some other part of dreamcoder.
        # boo camelCase
        self.outputDimensionality = 100

        self.conv1 = nn.Conv2d(1, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList([ResBlock(num_filters, num_filters)
            for i in range(num_blocks)])
        self.conv_1x1s = nn.ModuleList([OneByOneBlock(self.filters_1x1[i],
            self.filters_1x1[i+1]) for i in range(num_blocks)])

        self.lstm = nn.LSTM(self.intermediate_dim, self.lstm_hidden, 
            batch_first=False,
            bidirectional=False)
        self.linear = nn.Linear(self.lstm_hidden, self.outputDimensionality)

    def forward(self, x):
        x = torch.tensor(x).float()
        print('x shape: {}'.format(x.shape))
        x = x.view(-1, 1, self.input_dim)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        for block in self.basic_blocks:
            x = block(x)

        for c in self.conv_1x1s:
            x = c(x)

        print('x shape: {}'.format(x.shape))
        x = F.max_pool2d(x, kernel_size=x.size()[2:])

        hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
        x, hidden = lstm(x, hidden)

        x = F.relu(x)
        x = self.linear(x)
        return x


    #  we subclass nn.module, so this calls __call__, which calls forward()
    #  above
    def featuresOfTask(self, t): 
        # t.features is created when the task is made in makeArcTasks.py
        return self(t.features)

    # make tasks out of the program, to learn how the program operates (dreaming)
    def taskOfProgram(self, p, t):
        num_examples = random.choice(self.num_examples_list)

        def generate_sample():
            grid = self.sampler.sample()
            try: 
                out = p.evaluate([])(grid)
                example = (grid,), out
                return example
            except ValueError:
                return None

        examples = [generate_example() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples, features=make_features(examples))
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
            grid = ArcExample(np.repeat(np.random.randint(
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
            grid = ArcExample(np.random.randint(low=0,high=10,size=(1,3)))
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
            grid = ArcExample(grid)

            out = p.evaluate([])(grid)
            example = (grid,), out
            return example

        examples = [generate_example() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples, features=make_features(examples))
        return t

