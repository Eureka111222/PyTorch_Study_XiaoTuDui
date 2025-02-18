#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:06
# @Author : Yuzhao Li

import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid,ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    # 加载 CIFAR - 10 测试数据集，并将图像转换为 Tensor 类型
    dataset = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    # 创建数据加载器，设置每个批次包含 64 个样本
    dataloader = DataLoader(dataset, batch_size=64)
except Exception as e:
    print(f"数据集加载失败: {e}")
else:
    class Yuzhao(nn.Module):
        def __init__(self):
            super(Yuzhao, self).__init__()
            # self.relu=ReLU()
            # 定义 Sigmoid 激活函数层
            self.sigmoid = Sigmoid()

        def forward(self, input):
            # output=self.relu(input)
            # 使用 Sigmoid 激活函数处理输入
            output = self.sigmoid(input)
            return output

    # 创建模型实例
    yuzhao = Yuzhao()
    # 创建 SummaryWriter 对象，指定日志文件夹为 sigmoid
    writer = SummaryWriter("../logs")

    step = 0
    try:
        for data in dataloader:
            imgs, targets = data
            # 记录输入图像到 TensorBoard
            writer.add_images("input", imgs, step)
            # 对输入图像应用 Sigmoid 激活函数
            output = yuzhao(imgs)
            # 记录经过 Sigmoid 处理后的图像到 TensorBoard
            writer.add_images("sigmoid", output, step)
            step += 1
    except Exception as e:
        print(f"模型运行出错: {e}")
    finally:
        # 关闭 SummaryWriter 对象
        writer.close()
        print("end")