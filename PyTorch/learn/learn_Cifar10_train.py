#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:43
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
# 从 model.py 文件中导入所有定义的类和函数
# 这里假设 learn_savemodel.py 文件中定义了名为 Yuzhao 的神经网络类

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

# 准备数据集
# 加载 CIFAR - 10 训练数据集
# "../testdataset" 是数据集存储的路径
# train=True 表示加载训练集
# transform=torchvision.transforms.ToTensor() 将图像数据转换为 Tensor 类型
# download=True 若数据集不存在则进行下载
train_data = torchvision.datasets.CIFAR10("../testdataset", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
# 加载 CIFAR - 10 测试数据集
test_data = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

# len() 函数用于获取数据集的样本数量
train_data_size = len(train_data)
test_data_size = len(test_data)
# 格式化输出训练集和测试集的长度
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据集
# 使用 DataLoader 对训练数据集进行批量加载
# batch_size=64 表示每个批次包含 64 个样本
train_dataloader = DataLoader(train_data, batch_size=64)
# 对测试数据集进行批量加载
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
# 从 learn_savemodel.py 文件中引入的 Yuzhao 类创建一个神经网络实例
yuzhao = Yuzhao()

# 定义损失函数
# 使用交叉熵损失函数，常用于分类任务
loss_fn = nn.CrossEntropyLoss()

# 定义优化器的超参数
# 学习率设置为 0.01
learning_rate = 1e-2
# 使用随机梯度下降（SGD）优化器
# yuzhao.parameters() 获取神经网络的可训练参数
# lr=learning_rate 设置学习率
optimizer = torch.optim.SGD(yuzhao.parameters(), lr=learning_rate)

# 设置网络训练的一些参数
# 记录训练的总步数
total_train_step = 0
# 记录测试的总步数
total_test_step = 0
# 定义训练的总轮数
epoch = 10

# 开始训练过程
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))

    # 遍历训练数据加载器中的每个批次
    for data in train_dataloader:
        # 解包每个批次的数据，得到图像数据和对应的标签
        imgs, targets = data
        # 将图像数据输入到神经网络中进行前向传播，得到预测输出
        outputs = yuzhao(imgs)
        # 计算预测输出和真实标签之间的损失
        loss = loss_fn(outputs, targets)

        # 优化器进行参数更新的步骤
        # 清空优化器中之前计算的梯度
        optimizer.zero_grad()
        # 进行反向传播，计算损失函数关于网络参数的梯度
        loss.backward()
        # 根据计算得到的梯度更新网络参数
        optimizer.step()

        # 训练步数加 1
        total_train_step += 1
        # 打印当前训练步数和对应的损失值
        print("训练次数：{}，Loss:{}".format(total_train_step, loss))