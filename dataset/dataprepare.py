# -*- encoding: utf-8 -*-
'''
@File    :   dataprepare.py
@Time    :   2024/11/17 00:52:56
@Author  :   zsl
@Version :   1.0
@Contact :   2580333459@qq.com
@company :   Xidian University
'''

import scipy.io
import numpy as np
import os
from PIL import Image
import shutil

imagelabel_path = './dataset/imagelabels.mat'   #该地址为EXP1工程下的imagelabels.mat的相对地址
setid_path = './dataset/setid.mat'              #该地址为EXP1工程下的setid.mat的相对地址
pic_path = './102flowers/jpg/'                  #该地址为EXP1工程下的源数据图片的相对地址
img_size = (224,224)

labels = scipy.io.loadmat(imagelabel_path)
labels = np.array(labels['labels'][0]) - 1      #python处理数据从0开始
print("labels:", labels)

setid = scipy.io.loadmat(setid_path)
validation = np.array(setid['valid'][0]) - 1
np.random.shuffle(validation)                   #打乱数组顺序
train = np.array(setid['trnid'][0]) - 1
np.random.shuffle(train)
test = np.array(setid['tstid'][0]) - 1
np.random.shuffle(test)

flower_dir = list()                             #按顺序图像的地址list ['./102flowers/jpg/image_00001.jpg', './102flowers/jpg/image_00002.jpg'...]
for img in os.listdir(pic_path):
    flower_dir.append(os.path.join(pic_path, img))
    flower_dir.sort()
# print(flower_dir)

des_folder_train = "./dataset/train"            #该地址为新建的训练数据集文件夹的绝对地址
for tid in train:
    #打开图片并获取标签
    img = Image.open(flower_dir[tid])
    print(img)
    # print(flower_dir[tid])
    img = img.resize(img_size, Image.LANCZOS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_train, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)

des_folder_validation = "./dataset/valid"       #该地址为新建的验证数据集文件夹的绝对地址
for tid in validation:
    img = Image.open(flower_dir[tid])
    # print(flower_dir[tid])
    img = img.resize(img_size, Image.LANCZOS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_validation, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)

des_folder_test = "./dataset/train"#该地址为新建的测试数据集文件夹的绝对地址
for tid in test:
    img = Image.open(flower_dir[tid])
    # print(flower_dir[tid])
    img = img.resize(img_size, Image.LANCZOS)
    lable = labels[tid]
    # print(lable)
    path = flower_dir[tid]
    print("path:", path)
    base_path = os.path.basename(path)
    print("base_path:", base_path)
    classes = "c" + str(lable)
    class_path = os.path.join(des_folder_test, classes)
    # 判断结果
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    print("class_path:", class_path)
    despath = os.path.join(class_path, base_path)
    print("despath:", despath)
    img.save(despath)