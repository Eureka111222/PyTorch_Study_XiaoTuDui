#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:19
# @Author : Yuzhao Li
import torch
import torchvision

# 导入 PyTorch 的神经网络模块
from torch import nn
# 导入二维卷积层类，用于构建自定义神经网络
from torch.nn import Conv2d

# 加载预训练的 VGG16 模型，但不加载预训练权重
vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式 1：保存模型及参数
# torch.save() 函数用于将对象保存到指定的文件中
# 这里将整个 VGG16 模型（包括模型结构和参数）保存到名为 'vgg16_method1.pth' 的文件中
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式 2：只保存参数
# vgg16.state_dict() 返回一个包含模型所有参数的字典
# 将这个字典保存到名为 'vgg16_method2.pth' 的文件中
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 注意事项说明：
# 使用方法 1 保存自己写的网络时，在不同环境加载可能会有问题。
# 需要确保加载环境中能访问到该网络类的定义，比如把网络类代码复制过去，或者使用 'from model_save import *' 导入
class Yuzhao(nn.Module):
    # 自定义神经网络类 Yuzhao，继承自 nn.Module
    def __init__(self, *args, **kwargs) -> None:
        # 调用父类 nn.Module 的构造函数
        super().__init__(*args, **kwargs)
        # 定义一个二维卷积层，输入通道数为 3，输出通道数为 32，卷积核大小为 5x5
        self.conv1 = Conv2d(3, 32, 5)

    def forward(self, x):
        # 定义前向传播过程
        # 将输入数据 x 通过卷积层 self.conv1 进行处理
        x = self.conv1(x)
        return x

# 创建自定义神经网络 Yuzhao 的实例
yuzhao = Yuzhao()
# 使用保存方式 1 将自定义模型（包括结构和参数）保存到名为 'yuzhao_method1.pth' 的文件中
torch.save(yuzhao, "yuzhao_method1.pth")