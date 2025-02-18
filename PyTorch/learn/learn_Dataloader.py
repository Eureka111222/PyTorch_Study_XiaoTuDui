#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 9:28
# @Author : Yuzhao Li

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    # 加载 CIFAR - 10 测试数据集，并将图像数据转换为 Tensor 类型
    test_data = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor())
    # 创建数据加载器，将测试数据集按照 batch_size=64 进行批量加载
    # shuffle=False 表示不打乱数据顺序
    # num_workers=0 表示不使用多线程加载数据
    # drop_last=False 表示不丢弃最后一个不完整的批次
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
except Exception as e:
    print(f"数据集加载失败: {e}")
else:
    try:
        # 创建 SummaryWriter 对象，将日志文件保存到 ./logs/dataloader 目录下
        writer = SummaryWriter("../logs")
        # 进行 2 个 epoch 的数据遍历
        for epoch in range(2):
            step = 0
            # 遍历每个批次的数据
            for data in test_loader:
                # 取出一批图像和对应的标签
                imgs, targets = data
                # 将每个批次的图像写入 TensorBoard
                writer.add_images(f"test_dataloader_{epoch}", imgs, step)
                step += 1
    except Exception as e:
        print(f"数据写入 TensorBoard 失败: {e}")
    finally:
        # 关闭 SummaryWriter 对象，确保所有数据都被写入日志文件
        if 'writer' in locals():
            writer.close()
        print("end")
