#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 22:27
# @Author : Yuzhao Li

import torchvision
from torch.utils.tensorboard import SummaryWriter

# 定义数据转换操作
# 使用 Compose 组合多个数据转换操作，这里仅使用 ToTensor 将图像转换为 Tensor 类型
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

try:
    # 加载 CIFAR - 10 训练集
    # root: 数据集保存的根目录
    # train: True 表示加载训练集，False 表示加载测试集
    # transform: 应用的数据转换操作
    # download: 如果数据集不存在，是否下载
    train_set = torchvision.datasets.CIFAR10(root="../testdataset", train=True, transform=dataset_transform, download=True)
    # 加载 CIFAR - 10 测试集
    test_set = torchvision.datasets.CIFAR10(root="../testdataset", train=False, transform=dataset_transform, download=True)
except Exception as e:
    print(f"数据集加载失败: {e}")
else:
    # 创建 SummaryWriter 对象，将日志文件保存到 logs 目录下
    writer = SummaryWriter("../logs")

    try:
        # 从测试集中选取前 10 张图片进行可视化
        for i in range(10):
            # 获取第 i 张图片和对应的标签
            img, target = test_set[i]
            # 将图片写入 TensorBoard
            writer.add_image("test_set", img, i)
    except Exception as e:
        print(f"数据写入 TensorBoard 失败: {e}")
    finally:
        # 关闭 SummaryWriter 对象，确保所有数据都被写入日志文件
        writer.close()

    print("已结束")