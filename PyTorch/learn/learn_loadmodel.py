#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:22
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

# # 模型加载方法 1 - 对应保存方法 1
# # 此方法直接从文件中加载整个模型（包括模型结构和参数）
# # 加载保存的 VGG16 整个模型
# model1 = torch.load('./vgg16_method1.pth')
# # 打印加载后的模型结构
# print(model1)

# # # 加载方法 2
# # 这种方式只加载模型的参数
# # 直接加载保存的 VGG16 参数，此时得到的只是参数字典
# model2 = torch.load('./vgg16_method2.pth')
# # 打印参数字典
# print(model2)

# # 模型加载方法 2 - 对应保存方法 2
# # 若要完整加载网络，需先重新创建网络结构
# # 重新创建一个未加载预训练权重的 VGG16 模型
# vgg16 = torchvision.models.vgg16(weights=None)
# # 使用 load_state_dict 方法将保存的参数加载到新创建的 VGG16 模型中
# vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
# # 打印完整加载参数后的 VGG16 模型结构
# print(vgg16)


# 陷阱说明：
# # 若直接使用下面这行代码加载自定义模型，可能会出错
# # 因为在不同环境中加载自定义模型时，需要确保模型类的定义存在
# yuzhao1 = torch.load("yuzhao_method1.pth")
# print(yuzhao1)

# # 方法 1：手动定义模型类来解决加载自定义模型的问题
# # 定义自定义模型类 Yuzhao，继承自 nn.Module
# class Yuzhao(nn.Module):
#     # 类的初始化方法
#     def __init__(self, *args, **kwargs) -> None:
#         # 调用父类的初始化方法
#         super().__init__(*args, **kwargs)
#         # 定义一个二维卷积层，输入通道为 3，输出通道为 32，卷积核大小为 5
#         self.conv1 = Conv2d(3, 32, 5)
#
#     # 定义前向传播方法
#     def forward(self, x):
#         # 将输入数据 x 通过卷积层进行处理
#         x = self.conv1(x)
#         return x
#
# # 现在可以正常加载自定义模型了
# # 因为已经在当前环境中定义了 Yuzhao 类
# yuzhao2 = torch.load("yuzhao_method1.pth")
# # 打印加载后的自定义模型结构
# print(yuzhao2)


# 方法 2：直接把保存模型的文件引入
# 例如使用 from learn_savemodel import *
# 若在 learn_model_save 文件中定义了 Yuzhao 类，通过这种导入方式
# 可以在当前环境中使用该类，从而正确加载保存的自定义模型
from learn_savemodel import *
yuzhao3 = torch.load("yuzhao_method1.pth")
print(yuzhao3)