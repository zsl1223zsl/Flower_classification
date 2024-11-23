# -*- encoding: utf-8 -*-
'''
@File    :   flower_dataset.py
@Time    :   2024/11/17 00:53:09
@Author  :   zsl
@Version :   1.0
@Contact :   2580333459@qq.com
@company :   Xidian University
'''

import torch.utils.data as Data
from torchvision import transforms, models, datasets
import numpy as np
import random
import torch
import cv2
import os

DATA_ROOT = "./dataset"
train_dir = DATA_ROOT + '/train'
test_dir  = DATA_ROOT + '/test'
valid_dir = DATA_ROOT + '/valid'
 
# 进行数据增强操作
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def build_dataset():
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_ROOT, x), data_transforms[x]) for x in ['train', 'valid']}
    return image_datasets

