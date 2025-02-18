#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 9:43
# @Author : Yuzhao Li

import torch
import torch.nn.functional as F

# 定义一个 2 维的输入张量
# 这里模拟了一个 5x5 的图像矩阵，每个元素代表一个像素值
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 定义卷积核（算子）
# 卷积核是一个 3x3 的矩阵，用于在输入张量上进行卷积操作
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 改变输入张量和卷积核的形状
# 在 PyTorch 中，卷积层的输入和卷积核需要满足特定的形状要求
# 对于 2D 卷积，输入的形状应该是 (batch_size, in_channels, height, width)
# 这里 batch_size 为 1，表示只有一个样本；in_channels 为 1，表示输入只有一个通道
# 卷积核的形状应该是 (out_channels, in_channels, kernel_height, kernel_width)
# 这里 out_channels 为 1，表示输出只有一个通道；in_channels 为 1，表示输入只有一个通道
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# 打印输入张量和卷积核的形状，方便检查是否符合要求
print(input.shape)
print(kernel.shape)

# 使用 torch.nn.functional.conv2d 函数进行 2D 卷积操作
# input 是输入张量
# kernel 是卷积核
# stride=1 表示卷积核在输入张量上滑动的步长为 1
# 该函数会返回卷积操作后的输出张量
output = F.conv2d(input, kernel, stride=1)

# 打印卷积操作的输出结果
print(output)