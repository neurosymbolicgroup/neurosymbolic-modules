import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamcoder.domains.logo.main import Flatten


class ArcFeatureNN(nn.Module):
    special = "ARC"
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        super(ArcFeatureNN, self).__init__()

        self.recomputeTasks = False

        self.encoder = nn.Sequential(
            nn.Linear(100, 100),
            Flatten()
        )

        self.outputDimensionality = 256


    def forward(self, v):
        print('v: {}'.format(v))
        assert type(v) == list
        assert type(v[0]) == list
        for x in v:
            assert len(x) == 9

        return torch.from_numpy(np.array(v)).flatten()


    #  we subclass nn.module, so this calls __call__, which calls forward(self, v) above
    def featuresOfTask(self, t): 
        return self(t.features)


    def tasksOfPrograms(self, ps, types):
        for p in ps:
            print(p)
        print('tasks')
        assert False
        images = drawLogo(*ps, resolution=128)
        if len(ps) == 1: images = [images]
        tasks = []
        for i in images:
            if isinstance(i, str): tasks.append(None)
            else:
                t = Task("Helm", arrow(turtle,turtle), [])
                t.highresolution = i
                tasks.append(t)
        return tasks        

    def taskOfProgram(self, p, t):
        return None
