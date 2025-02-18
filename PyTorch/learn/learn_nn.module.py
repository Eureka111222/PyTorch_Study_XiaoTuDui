#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 9:35
# @Author : Yuzhao Li

import torch
from torch import nn

# 创建自己的网络类，继承自 nn.Module
# nn.Module 是 PyTorch 中所有神经网络模块的基类，自定义的神经网络模型通常需要继承这个类
# 通过继承 nn.Module，我们可以方便地利用 PyTorch 提供的自动求导、参数管理等功能
class Yuzhao(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        # 调用父类 nn.Module 的构造函数，确保父类的初始化逻辑被正确执行
        # 这样可以继承父类的属性和方法，如参数管理、模块注册等
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # 定义前向传播过程，这是自定义神经网络模型必须实现的方法
        # 前向传播是指数据从输入层经过一系列的计算和变换得到输出的过程
        # 在这个简单的例子中，我们对输入进行了一个简单的加 1 操作
        output = input + 1
        return output

# 创建 Yuzhao 类的实例
yuzhao = Yuzhao()

# 创建一个包含单个元素 1.0 的张量作为输入
x = torch.tensor(1.0)

# 调用 yuzhao 实例，实际上会调用其 forward 方法进行前向传播计算
# 在 PyTorch 中，调用一个继承自 nn.Module 的实例时，会自动调用其 forward 方法
output = yuzhao(x)

# 打印前向传播的输出结果
print(output)