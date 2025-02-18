#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:04
# @Author : Yuzhao Li
import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

# 由于 ImageNet 数据集版权问题，通常需要手动下载和配置，这里注释掉
# train_data = torchvision.datasets.ImageNet("../data_ImageNet", split='train', transform=torchvision.transforms.ToTensor(), download=True)

# 创建不加载预训练权重的 VGG16 模型
vgg16_false = torchvision.models.vgg16(weights=None)
# 创建加载预训练权重的 VGG16 模型
# vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

# 查看不加载预训练权重的 VGG16 模型架构
print("不加载预训练权重的 VGG16 模型架构：")
print(vgg16_false)
print("\n")

#在整个网络后加模型
# 迁移学习，在 vgg16 后面加一个线性层（这里注释掉示例代码，仅展示思路）
# vgg16_false.add_module('add_linear', nn.Linear(1000, 10))
# print("在整个模型后面添加线性层后的 VGG16 模型架构：")
# print(vgg16_false)

#再VGG16写好的模块中加模型
# 在 vgg16 的 classifier 模块后面加一个线性层（这里注释掉示例代码，仅展示思路）
# vgg16_false.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print("在 classifier 模块后面添加线性层后的 VGG16 模型架构：")
# print(vgg16_false)

#修改指定的网络层
# 修改 vgg16 模型的 classifier 模块中的第 6 个线性层
vgg16_false.classifier[6] = nn.Linear(4096, 100)
print("修改 classifier 模块中最后一个线性层后的 VGG16 模型架构：")
print(vgg16_false)

print("ok!")