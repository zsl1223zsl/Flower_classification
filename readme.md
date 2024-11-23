---
title: 	基于ResNet-18模型实现花卉图像分类
date: 	2024-11-14
author: zsl
email:	2580333459@qq.com

---



# 基于 ResNet-18 模型实现花卉图像分类

本课程实验是基于西安电子科技大学21级通信工程教改班的大四课程————智能计算系统。

## 实验要求

一、实验描述

​	基于ResNet-18模型， 使用Tensorflow/PyTorch，训练一个花卉图像分类器，该分类器可以将花卉图像分为不同的花卉种类。

二、实验要求

1. 基于python语言实现，深度学习编程库以TensorFlow或Pytorch均可；
2. 理解深度学习模型的训练步骤、训练过程、在分类问题中的应用；
3. 花卉数据下载：102 Category Flower Dataset，将数据分为训练集和测试集；
4. 训练的分类器对测试集的测试精度达到90%以上；
5. 实验报告内容包括：
   - 简洁介绍实验环境部署情况 
   - 主要的实验步骤
   - 实验结果(迭代损失函数趋势、对训练集和测试集的分类精度，建议绘图)
   - 实验分析或者改进（可选）

三、提交报告要求

1. pdf或word格式， 文件命名格式：姓名+学号+实验一
2. 上传到学在西电平台的“实验一”文件夹下
3. 报告上传截止日期：2024.11.30

## 实现流程

一、下载数据集

[aiaaee/102-Flower-Dataset: 102Flower is an image classification dataset consisting of 102 flower categories (github.com)](https://github.com/aiaaee/102-Flower-Dataset?tab=readme-ov-file)

官方除了数据级还给了imagelabels.mat与setid.mat：

- imagelabels.mat：一个字段，根据图片的顺序标注每张图片的类别，图片总共有8189张。

- setid.mat：三个字段，划分好的训练(1020)、测试(6149)、验证集(1020)索引。

- 由于实验要求把数据分为训练集和测试集，工程将原测试集并入训练集，并把验证集当作测试集。

首先解压102flowers.tgz（默认路径即可），运行dataset中的dataprepare.py文件处理数据

```
python ./dataset/dataprepare.py
```

二、查看Resnet-18网络结构

下载Resnet论文[1512.03385 (arxiv.org)](https://arxiv.org/pdf/1512.03385)

网络上有很多Resnet的开源代码可以借鉴，在此感谢各个开源。

三、使用pytorch编写Resnet代码

安装环境

```
pip install -r requirements.txt
```

配置好相应的pytorch环境之后，可以直接运行train.py，就可以从零开始训练网络。

```
python train.py
```

查看损失函数趋势

```
tensorboard --logdir=./logs
```

文件树

```
exp1
├─ 102flowers
│  └─ jpg
│     ├─ image_00001.jpg
│     ├─ ...
│     └─ image_08189.jpg
├─ 102flowers.tgz
├─ dataset
│  ├─ dataprepare.py
│  ├─ imagelabels.mat
│  ├─ setid.mat
│  ├─ valid
│  │  ├─ c0
│  │  │  ├─ image_06734.jpg
│  │  │  ├  ...
│  │  │  └─ image_06772.jpg
│  │  ├─ c1
│  │  │  ...
│  │  └─ c101
│  └─ train
│     ├─ c0
│     │  ├─ image_06734.jpg
│     │  └─ image_06772.jpg
│     ├─ c1
│     │  ...
│     └─ c101
├─ model_par
│  ├─ fine_train.pth
│  └─ train.pth
├─ readme
├─ Resnet.pdf
├─ flower_dataset.py
├─ net.py
├─ train.py
└─ logs

```

