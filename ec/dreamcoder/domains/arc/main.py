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


class Sampler():
    def __init__(self, grid):
        self.grid = grid

    def sample(self):
        l = [random.randint(1, MAX_COLOR) for i in range(0, MAX_COLOR+1)]
        new_grid = self.grid.transform({i: l[i] for i in range(len(l))}) 
        return new_grid

class ArcFeatureNN(nn.Module):
    special = "ARC"
    
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

        examples = [generate_e_xample() for _ in range(num_examples)]
        if None in examples:
            return None
        t = Task("Helm", t, examples, features=make_features(examples))
        return t

