#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:15
# @Author : Yuzhao Li
import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载 CIFAR - 10 测试数据集
# root 参数指定数据集存储的路径为 "../testdataset"
# train=False 表示加载的是测试集
# transform=torchvision.transforms.ToTensor() 表示将图像数据转换为 Tensor 类型
# download=True 表示如果数据集不存在则进行下载
dataset = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 创建 DataLoader 对象，用于批量加载数据集
# batch_size=64 表示每个批次包含 64 个样本
dataloader = DataLoader(dataset, batch_size=64)

# 定义自定义的神经网络类，继承自 nn.Module
class Yuzhao(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)
        # 定义一个全连接层（线性层）
        # 输入特征的数量为 196608
        # 输出特征的数量为 10
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        # 定义前向传播过程
        # 将输入数据通过线性层进行变换
        output = self.linear1(input)
        return output

# 创建自定义神经网络的实例
yuzhao = Yuzhao()

# 遍历数据加载器中的每个批次的数据
for data in dataloader:
    # 解包每个批次的数据，得到图像数据和对应的标签
    imgs, targets = data
    # 打印图像数据的形状
    print("图像原始维度：")
    print(imgs.shape)
    # 尝试将图像数据展平，但此处代码只是注释掉的调用，未实际执行
    # torch.flatten(imgs)
    # 将图像数据进行形状重塑
    # 这里使用 -1 让 PyTorch 自动计算该维度的大小
    # 最终将数据重塑为形状 (1, 1, 1, -1)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    # 打印重塑后数据的形状
    # print(output.shape)
    # 将重塑后的数据输入到自定义神经网络中进行前向传播
    output = yuzhao(output)
    print("图像经过Linear后维度")
    # 打印经过神经网络处理后输出数据的形状
    print(output.shape)