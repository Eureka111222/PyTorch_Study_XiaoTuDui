#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:30
# @Author : Yuzhao Li
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

# 定义输入张量，模拟模型的预测输出
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# 定义目标张量，即真实的标签值
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# 调整输入张量的形状，使其符合四维张量的格式 (batch_size, channels, height, width)
# 这里 batch_size 为 1，channels 为 1，height 为 1，width 为 3
inputs = torch.reshape(inputs, (1, 1, 1, 3))
# 同样调整目标张量的形状
targets = torch.reshape(targets, (1, 1, 1, 3))

# 创建 L1 损失函数对象
# L1 损失（也称为平均绝对误差，MAE）是预测值与真实值之间绝对误差的平均值
# 默认情况下，reduction 参数为 'mean'，表示取平均值
# 这里将 reduction 参数设置为 'sum'，表示对所有误差求和
# loss = L1Loss()  # 默认取平均值
loss = L1Loss(reduction="sum")
# 计算输入和目标之间的 L1 损失
result = loss(inputs, targets)
# 打印 L1 损失的计算结果
print(result)

# 创建均方误差（MSE）损失函数对象
# MSE 损失是预测值与真实值之间误差平方的平均值
loss_mse = MSELoss()
# 计算输入和目标之间的 MSE 损失
result_mse = loss_mse(inputs, targets)
# 打印 MSE 损失的计算结果
print(result_mse)

# 定义一个新的输入张量，模拟模型的预测输出（通常是未经过 softmax 处理的 logits）
x = torch.tensor([0.1, 0.2, 0.3])
# 定义对应的目标标签，这里表示真实类别为 1
y = torch.tensor([1])
# 调整输入张量的形状，使其符合二维张量的格式 (batch_size, num_classes)
# 这里 batch_size 为 1，num_classes 为 3
x = torch.reshape(x, (1, 3))
# 创建交叉熵损失函数对象
# 交叉熵损失常用于分类问题，它衡量的是两个概率分布之间的差异
loss_cross = CrossEntropyLoss()
# 计算输入和目标之间的交叉熵损失
# 交叉熵损失的计算公式为：loss = -x[class] + log(求和(exp(xi)))
result_cross = loss_cross(x, y)
# 打印交叉熵损失的计算结果
print(result_cross)