#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 12:30
# @Author : Yuzhao Li
import torch
import torchvision
from torch import argmax, nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../testdataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 格式化输出
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Yuzhao(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 使用 sequential
        self.model1 = nn.Sequential(
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

if torch.cuda.is_available():
    yuzhao = yuzhao.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(yuzhao.parameters(), lr=learning_rate)

# 设置网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# tensorboard 记录有关信息
writer = SummaryWriter("../logs/train")

# 训练
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))
    # 设置模型为训练模式
    yuzhao.train()
    for data in train_dataloader:
        start_time = time.time()  # 记录每个批次开始的时间
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = yuzhao(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器调参
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        end_time = time.time()
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
            print("用时:{}".format(end_time - start_time))
            writer.add_scalar("train_loss", loss, total_train_step)

    # 使用测试集测试一下模型
    # 设置模型为评估模式
    yuzhao.eval()
    total_test_loss = 0
    total_acc_num = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = yuzhao(imgs)
            loss = loss_fn(outputs, targets)

            # 计算当前这组测试的正确测试数
            acc_num = (outputs.argmax(1) == targets).sum()
            # 在这一轮总的测试中加上这组的正确数
            total_acc_num += acc_num
            total_test_loss += loss

        total_accuracy = total_acc_num / test_data_size
        print("整体测试集上的 Loss:{}".format(total_test_loss))
        print("整体测试集上的正确率:{}".format(total_accuracy))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
        total_test_step += 1

    # 保存模型
    # torch.save(yuzhao, "yuzhao_{}.pth".format(i + 1))

writer.close()