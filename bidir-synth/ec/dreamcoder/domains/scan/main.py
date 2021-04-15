import numpy as np
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

from ec.dreamcoder.task import Task

class ScanNet(nn.Module):
    
    def __init__(self, tasks, testingTasks=[], cuda=False):
        super().__init__()

        # maybe this should be True, See recognition.py line 908
        self.recomputeTasks = False

        # map word to index 
        input_strs = [t.examples[0][0][0] for t in tasks]
        words = set(w for input_str in input_strs for w in input_str.split(' '))
        self.word_embedding = dict(zip(words, range(len(words))))
        self.input_size = len(self.word_embedding)
        self.hidden_size = 64
        # need to keep this named this for some other part of dreamcoder.
        self.outputDimensionality = self.hidden_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                bidirectional=False)


    def forward(self, x):
        # (seq_len, input_size)
        x = x.view(-1, 1, self.input_size)
        # input shape: (seq_len, batch, input_size)
        # output is the features from last seq item, shape: (1, batch, hidden_size)
        _, (out, _) = self.lstm(x)

        # shape: [hidden_size]
        return out.view(-1)


    def make_features(self, examples):
        assert len(examples) == 1
        input_str = examples[0][0][0]
        words = input_str.split(' ')
        vec = torch.tensor([self.word_embedding[w] for w in words])
        vec = F.one_hot(vec, num_classes=self.input_size)
        vec = vec.to(torch.float32)
        return vec
        

    def featuresOfTask(self, t):
        return self(self.make_features(t.examples))

    

