#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:39
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

try:
    # 加载 CIFAR - 10 测试数据集，并将图像数据转换为 Tensor 类型
    dataset = torchvision.datasets.CIFAR10("../testdataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    # 创建数据加载器，每个批次包含 64 个样本
    dataloader = DataLoader(dataset, batch_size=64)
except Exception as e:
    print(f"数据集加载失败: {e}")
else:
    # 定义自定义神经网络类
    class Yuzhao(nn.Module):
        def __init__(self):
            super(Yuzhao, self).__init__()
            self.model1 = Sequential(
                Conv2d(3, 32, 5, padding=2),
                MaxPool2d(2),
                Conv2d(32, 32, 5, padding=2),
                MaxPool2d(2),
                Conv2d(32, 64, 5, padding=2),
                MaxPool2d(2),
                Flatten(),
                Linear(1024, 64),
                Linear(64, 10)
            )

        def forward(self, x):
            x = self.model1(x)
            return x

    # yuzhao = Yuzhao()
    # loss_cross = nn.CrossEntropyLoss()
    # for data in dataloader:
    #     imgs, targets = data
    #     outputs = yuzhao(imgs)
    #     # print(outputs)
    #     # print(targets)
    #     # 使用loss函数
    #     results = loss_cross(outputs, targets)
    #     # 反向传播计算梯度
    #     grad = results.backward()
    #     print(results)


    # 创建模型实例
    yuzhao = Yuzhao()
    # 创建交叉熵损失函数
    loss_cross = nn.CrossEntropyLoss()
    # 定义优化器，使用随机梯度下降（SGD）
    optimizer = torch.optim.SGD(yuzhao.parameters(), lr=0.001)

    for i in range(10):
        for data in dataloader:
            imgs, targets = data
            # 前向传播
            outputs = yuzhao(imgs)
            # 计算损失
            results = loss_cross(outputs, targets)

            # # 清空优化器中的梯度
            # optimizer.zero_grad()
            # 反向传播计算梯度
            results.backward()
            # 根据梯度更新模型参数
            optimizer.step()

            print(results.item())

    print("训练结束")