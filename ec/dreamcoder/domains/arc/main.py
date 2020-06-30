import numpy as np
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.task import Task
from dreamcoder.domains.logo.main import Flatten
from dreamcoder.domains.arc.arcPrimitives import ArcExample
from dreamcoder.domains.arc.makeArcTasks import make_features



class ArcFeatureNN(nn.Module):
    special = "ARC"
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super(ArcFeatureNN, self).__init__()

        self.recomputeTasks = False

        self.num_examples_list = [len(t.examples) for t in tasks]

        self.encoder = nn.Sequential(
            nn.Linear(100, 100),
            Flatten()
        )

        self.outputDimensionality = 18


    def forward(self, v):
        v = np.array(v)
        assert v.shape == (2,3,3), "not the shape we were planning on using"
        return torch.from_numpy(v.flatten()).float()


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
            input_grid = ArcExample(np.repeat(np.random.randint(
                    low=0, high=10, size=(1,3)), 3, axis=0))
            out = p.evaluate([])(input_grid)
            example = (input_grid,), out
            return example

        examples = [generate_example() for _ in range(num_examples)]
        t = Task("Helm", t, examples, features=make_features(examples))
        return t
