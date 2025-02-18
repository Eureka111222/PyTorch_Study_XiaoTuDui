#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 21:05
# @Author : Yuzhao Li

from torch.utils.data import  Dataset
from PIL import Image
import cv2
import os

class MyData(Dataset):
    #将输入的路径拼接并转成一个list
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir  #获取根目录
        self.label_dir=label_dir #子目录
        self.path=os.path.join(self.root_dir,self.label_dir) #将根目录与子目录拼接
        self.img_path=os.listdir(self.path) #将这个文件夹下的东西变成一个list

    #获取具体的一个数据->在训练时一般有这个函数
    def __getitem__(self, idx):
        img_name=self.img_path[idx] #获取图片名称
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    #获取数据集长度
    def __len__(self):
        return len(self.img_path)

#测试
root_dir= "../dataset/train"
ants_label="ants"
bees_label="bees"
ants_dataset=MyData(root_dir,ants_label) #构建具体的数据集
bees_dataset=MyData(root_dir,bees_label)

# #展示第一张蚂蚁数据集的图片
# img,label=ants_dataset[0]
# print(label)
# img.show()

# 将两个数据集合并为一个
train_dataset=ants_dataset+bees_dataset
print(len(train_dataset))

# 展示数据集第125张图像，因为ants只有124张图片，所以第125张图片就是bees的第一张，是一个蜜蜂
img,label=train_dataset[124]
print(label)
img.show()


