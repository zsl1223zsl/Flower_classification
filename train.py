# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/11/17 00:52:44
@Author  :   zsl
@Version :   1.0
@Contact :   2580333459@qq.com
@company :   Xidian University
'''

from net import ResNet,ResidualBlock
from flower_dataset import build_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import datetime
import time

from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter

# set global parameter
use_pretrainpth_flag = False # whether use pretrain parameters
train_flag = False
test_flag = False
checkpoint_path = './model_par'

# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameter
EPOCH = 100
pre_epoch = 0
BATCH_SIZE = 4
LR = 0.01
optim_step = 4
optim_gamma = 0.1

# tensorboard loss
# use command tensorboard --logdir=./ to check
Loss_list = SummaryWriter('./logs/train_logs')
losslisti = 0
test_Loss_list = SummaryWriter('./logs/test_logs') # only record every epoches average test loss
test_losslisti = 0
train_Acc_list = SummaryWriter('./logs/train_Acc_logs')
train_acclisti = 0
test_Acc_list = SummaryWriter('./logs/test_Acc_logs')
test_acclisti = 0

# make dataset and dataloader
image_datasets = build_dataset()
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
trainloader = dataloaders['train']
testloader = dataloaders['valid']

# time
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

# define ResNet18
net = ResNet(ResidualBlock).to(device)
if use_pretrainpth_flag:
    ckpt_path=checkpoint_path+"/train.pth"
    net = torch.load(ckpt_path)
    LR = 1e-6

# Loss and optimizer
# define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=LR)
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)

prev_time = time.time()
# train
for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        # prepare dataset
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # forward & backward
        outputs = net(inputs)
        _, output_label = torch.max(outputs,1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Determine approximate time left
        batches_done = epoch * len(trainloader) + i
        batches_left = EPOCH * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        # print ac & loss in each batch
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | ETA: %.10s' 
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, time_left))
        # save loss value every 10 iters
        if i % 10 == 0:
            Loss_list.add_scalar("loss",loss.item(),losslisti)
            losslisti += 1
            train_Acc_list.add_scalar("train_Acc",correct / total,train_acclisti)
            train_acclisti += 1
        if 100. * correct / total > 90:
            train_flag = True

    if epoch > 50:
        scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

    if use_pretrainpth_flag:
        torch.save(net, os.path.join(checkpoint_path+"/fine_train_1.pth"))
    else:
        torch.save(net, os.path.join(checkpoint_path+"/train.pth"))
    #get the ac with testdataset in each epoch
    print('Waiting Test...')
    with torch.no_grad():
        correct = 0
        total = 0
        sum_loss = 0.0
        for i, data in enumerate(testloader,0):
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('[Test\'s parameter] Loss: %.03f |Acc: %.3f%%' 
              % (sum_loss / (i + 1), 100. * correct / total))
        test_Loss_list.add_scalar("loss",sum_loss / (i + 1),test_losslisti)
        test_losslisti += 1
        test_Acc_list.add_scalar("train_Acc",correct / total,test_acclisti)
        test_acclisti += 1
        if 100. * correct / total > 90:
            test_flag = True

    if test_flag & train_flag:
        break
    

print('Train has finished, total epoch is %d' % EPOCH)
Loss_list.close()
test_Loss_list.close()
train_Acc_list.close()
test_Acc_list.close()