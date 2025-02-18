#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:47
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
import time

try:
    # 加载 CIFAR - 10 训练数据集，并将图像转换为 Tensor 类型
    train_dataset = torchvision.datasets.CIFAR10("../testdataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    # 加载 CIFAR - 10 测试数据集
    test_dataset = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
except Exception as e:
    print(f"数据集加载失败: {e}")
else:
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

    yuzhao = Yuzhao()
    loss_cross = nn.CrossEntropyLoss()
    # 创建优化器，使用更具描述性的名称
    optimizer = torch.optim.SGD(yuzhao.parameters(), lr=0.01)

    epochs = 20
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        yuzhao.train()  # 设置模型为训练模式
        for data in train_dataloader:
            imgs, targets = data
            outputs = yuzhao(imgs)
            # 计算损失
            results = loss_cross(outputs, targets)

            # 清空梯度
            optimizer.zero_grad()
            #反向传播
            results.backward()
            optimizer.step()

            running_loss += results.item()

        end_time = time.time()
        # 计算每个 epoch 的训练时间
        train_time = end_time - start_time

        # 打印每个 epoch 的损失和训练时间
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader):.4f}, Time: {train_time:.2f}s')

        # 在每个 epoch 结束后进行模型评估
        yuzhao.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                outputs = yuzhao(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # 打印每个 epoch 的测试准确率
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    print("训练结束")