#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:22
# @Author : Yuzhao Li
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


# 定义自定义的神经网络类 Yuzhao，继承自 nn.Module
class Yuzhao(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的构造函数，确保父类的初始化逻辑被执行
        super().__init__(*args, **kwargs)

        # 原本逐个定义网络层的方式，被注释掉了
        # self.conv1=Conv2d(3,32,5,padding=2)  # 第一个卷积层，输入通道 3，输出通道 32，卷积核大小 5，填充 2
        # self.maxpool1=MaxPool2d(2)  # 第一个最大池化层，池化核大小 2
        # self.conv2=Conv2d(32,32,5,padding=2)  # 第二个卷积层，输入通道 32，输出通道 32，卷积核大小 5，填充 2
        # self.maxpool2=MaxPool2d(2)  # 第二个最大池化层，池化核大小 2
        # self.conv3=Conv2d(32,64,5,padding=2)  # 第三个卷积层，输入通道 32，输出通道 64，卷积核大小 5，填充 2
        # self.maxpool3=MaxPool2d(2)  # 第三个最大池化层，池化核大小 2
        # self.flatten=Flatten()  # 用于将多维张量展平为一维
        # self.linear1=Linear(1024,64)  # 第一个全连接层，输入特征 1024，输出特征 64
        # self.linear2=Linear(64,10)  # 第二个全连接层，输入特征 64，输出特征 10

        # 使用 Sequential 容器来组织网络层
        # Sequential 可以按顺序依次执行其中的网络层，简化代码结构
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 第一个卷积层
            MaxPool2d(2),  # 第一个最大池化层
            Conv2d(32, 32, 5, padding=2),  # 第二个卷积层
            MaxPool2d(2),  # 第二个最大池化层
            Conv2d(32, 64, 5, padding=2),  # 第三个卷积层
            MaxPool2d(2),  # 第三个最大池化层
            Flatten(),  # 展平层
            Linear(1024, 64),  # 第一个全连接层
            Linear(64, 10)  # 第二个全连接层
        )

    def forward(self, x):
        # 原本逐个调用网络层的前向传播方式，被注释掉了
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)

        # 直接通过 Sequential 容器进行前向传播
        x = self.model1(x)

        return x

# 创建 Yuzhao 类的实例
yuzhao = Yuzhao()
# 打印模型的结构信息
print(yuzhao)

# 构造一个测试输入数据
# 形状为 (64, 3, 32, 32)，表示批量大小为 64，通道数为 3，高度和宽度均为 32 的输入张量
input = torch.ones(64, 3, 32, 32)
# 将输入数据传入模型进行前向传播
output = yuzhao(input)
# 打印输出数据的形状
print(output.shape)

# 创建一个 SummaryWriter 对象，用于将模型结构写入 TensorBoard 日志
writer = SummaryWriter("../logs")
# 将模型的计算图添加到 TensorBoard 中，方便可视化模型结构
writer.add_graph(yuzhao, input)
# 关闭 SummaryWriter 对象，确保所有数据都被写入日志文件
writer.close()