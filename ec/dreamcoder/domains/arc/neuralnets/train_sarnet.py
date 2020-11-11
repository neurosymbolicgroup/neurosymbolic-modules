import numpy as np
import scipy.io as sio
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import UNetSAR
from sar_data_generator import SARDataGenerator
from sarnet_config import SARConfig as conf

import pickle
import os
import matplotlib.pyplot as plt

if conf.train_regime == 'fixed':
    if hasattr(conf, 'trainset_path') is False:
        template = sio.loadmat(conf.template_path)['st']
        missing_rate = 0.50
        datagen = SARDataGenerator(template,
                                   dim=conf.dim,
                                   max_sparse=conf.max_sparse,
                                   sparsity_pattern=conf.sparsity_pattern,
                                   support_dist=conf.support_dist,
                                   scale=conf.scale,
                                   num_clusters=conf.num_clusters,
                                   normalize_signals=conf.normalize_signals)
        signals, measurements = datagen.generate_and_save_dataset(conf.trainset_size, missing_rate=missing_rate)
        signals, measurements = torch.Tensor(signals), torch.Tensor(measurements)
    else:
        with open(conf.trainset_path,'rb') as f:
            dataset = pickle.load(f)
            signals, measurements = torch.Tensor(dataset['signals']), torch.Tensor(dataset['measurements'])

    trainloader = torch.utils.data.DataLoader(zip(signals, measurements), batch_size=conf.batch_size)
    print('Loaded Training Dataset')

elif conf.train_regime == 'online':
    template = sio.loadmat(conf.template_path)['st']
    missing_rate = 0.50
    datagen = SARDataGenerator(template,
                               dim=conf.dim,
                               max_sparse=conf.max_sparse,
                               sparsity_pattern=conf.sparsity_pattern,
                               support_dist=conf.support_dist,
                               scale=conf.scale,
                               num_clusters=conf.num_clusters,
                               normalize_signals=conf.normalize_signals)
    print('Created Training Simulator')

if conf.criterion == 'l1':
    criterion = nn.L1Loss()
elif conf.criterion == 'mse':
    criterion = nn.MSELoss()
elif conf.criterion == 'smoothl1':
    criterion = nn.SmoothL1Loss()
else:
    raise Exception('Invalid Criterion, defaulting to L1')
    criterion = nn.L1Loss()

net = UNetSAR()
if torch.cuda.is_available() and conf.use_gpu:
    net = net.cuda()
    criterion = criterion.cuda()

print('Loaded Network')

losses = []
optimizer = optim.Adam(net.parameters(), lr=conf.learning_rate)
if conf.lr_schedule:
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          conf.lr_schedule_interval,
                                          gamma=conf.lr_schedule_gamma)

if conf.train_regime == 'fixed':
    for epoch in range(conf.max_epochs):
        running_loss = 0.0

        for i,data in enumerate(trainloader,0):
            labels, inputs = data
            if torch.cuda.is_available() and conf.use_gpu:
                labels, inputs = Variable(labels.cuda()), Variable(inputs.cuda())

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            running_loss += np.float(loss.data.cpu().numpy())
            if conf.verbose and (i+1) % conf.print_every == 0:
                print('[epoch: %d, iter: %d] loss: %.5f' % (epoch+1, i+1, running_loss/conf.print_every))
                running_loss = 0.0

            if conf.verbose and i==0 and epoch==0:
                print('[epoch: %d, iter: %d] loss: %.5f' % (epoch+1, i+1, running_loss))

        if conf.lr_schedule:
            scheduler.step()
        if (epoch+1) % conf.save_every == 0 or epoch+1==conf.max_epochs:
            if conf.save_training_plot:
                fig = plt.figure(figsize=(16, 9));
                plt.title("Training loss")
                plt.plot(losses)
                plt.xlabel("Iteration")
                plt.ylabel(conf.criterion)
                plt.tight_layout()
                plt.savefig(os.path.join(conf.figure_dir, 'UNet_%s_%s_epoch%d_maxsparse%d.png'%(conf.train_regime, conf.criterion, epoch+1, conf.max_sparse)))

            torch.save(net.cpu(),os.path.join(conf.save_dir, 'UNet_%s_%s_epoch%d_maxsparse%d.pkl'%(conf.train_regime, conf.criterion, epoch+1, conf.max_sparse)))
            if torch.cuda.is_available() and conf.use_gpu:
                net = net.cuda()

elif conf.train_regime == 'online':
    for epoch in range(conf.max_epochs):
        running_loss = 0.0

        for i in range(conf.iters_per_epoch):
            labels, inputs = datagen.generate_batch(conf.batch_size, missing_rate=missing_rate)
            labels, inputs = torch.Tensor(labels). torch.Tensor(inputs)
            if torch.cuda.is_available() and conf.use_gpu:
                labels, inputs = Variable(labels.cuda()), Variable(inputs.cuda())

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            running_loss += np.float(loss.data.cpu().numpy())
            if conf.verbose and (i+1) % conf.print_every == 0:
                print('[epoch: %d, iter: %d] loss: %.5f' % (epoch+1, i+1, running_loss/conf.print_every))
                running_loss = 0.0

            if conf.verbose and i==0 and epoch==0:
                print('[epoch: %d, iter: %d] loss: %.5f' % (epoch+1, i+1, running_loss))

        if conf.lr_schedule:
            scheduler.step()
        if (epoch+1) % conf.save_every == 0 or epoch+1==conf.max_epochs:
            if conf.save_training_plot:
                fig = plt.figure(figsize=(16, 9));
                plt.title("Training loss")
                plt.plot(losses)
                plt.xlabel("Iteration")
                plt.ylabel(conf.criterion)
                plt.tight_layout()
                plt.savefig(os.path.join(conf.figure_dir, 'UNet_%s_%s_epoch%d_maxsparse%d.png'%(conf.train_regime, conf.criterion, epoch+1, conf.max_sparse)))

            torch.save(net.cpu(),os.path.join(conf.save_dir, 'UNet_%s_%s_epoch%d_maxsparse%d.pkl'%(conf.train_regime, conf.criterion, epoch+1, conf.max_sparse)))
            if torch.cuda.is_available() and conf.use_gpu:
                net = net.cuda()
