#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 9:55
# @Author : Yuzhao Li
import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    # 加载 CIFAR - 10 测试数据集，并将图像数据转换为 Tensor 类型
    dataset = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    # 创建数据加载器，每个批次包含 64 个样本
    dataloader = DataLoader(dataset, batch_size=64)
except Exception as e:
    print(f"数据集加载失败: {e}")
else:
    # 定义自定义神经网络类
    class Yuzhao(nn.Module):
        def __init__(self):
            super(Yuzhao, self).__init__()
            self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

        def forward(self, input):
            output = self.maxpool1(input)
            return output

    # 创建模型实例
    yuzhao = Yuzhao()

    # 创建 TensorBoard 写入器
    writer = SummaryWriter("../logs")

    step = 0
    try:
        # 遍历数据加载器中的每个批次数据
        for data in dataloader:
            imgs, targets = data
            # 写入输入图像到 TensorBoard
            writer.add_images("input", imgs, step)
            # 将图像数据输入到模型中进行最大池化操作
            output = yuzhao(imgs)
            # 写入最大池化后的输出图像到 TensorBoard
            writer.add_images("maxpool", output, step)
            step += 1
    except Exception as e:
        print(f"模型运行出错: {e}")
    finally:
        # 关闭 TensorBoard 写入器
        writer.close()
        print("end")