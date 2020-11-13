import numpy as np
import scipy.io as sio
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import UNetSAR

import pickle
import os
import matplotlib.pyplot as plt

# config = {
#     "batch_size": 2,
#     "learning_rate": .1,
#     "max_epochs": 10
# }

signals = np.random.rand(5,1,1000) # 5 training samples with 1 channel in 1000-dim space
measurements =np.random.rand(1000,5)  # 5 training examples classified as a 0 or 1
signals, measurements = torch.Tensor(signals), torch.Tensor(measurements)

# to help batch it 
# trainloader = torch.utils.data.DataLoader(zip(signals, measurements), batch_size=config["batch_size"])
# print('Loaded Training Dataset')

net = UNetSAR()
out = net(signals)
#print(out.shape)

# plot the input and plot the output
# plt.plot(signals[0,0,:]); plt.show()
# plt.plot(out.detach().numpy()[0,0,:]); plt.show()

