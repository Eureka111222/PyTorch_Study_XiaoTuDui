#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 21:18
# @Author : Yuzhao Li

# 从 torch 库的 utils.tensorboard 模块中导入 SummaryWriter 类
# SummaryWriter 用于将数据写入 TensorBoard 所需的日志文件，方便后续可视化
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 创建一个 SummaryWriter 对象，指定日志文件的保存目录为 "logs"
# 后续写入的数据（如图像、标量等）都会存储在该目录下的日志文件中
writer = SummaryWriter("../logs")

# 定义要读取的图像文件的路径
# 这里假设在当前工作目录下存在 "dataset/train/bees" 文件夹，且其中包含指定名称的图像文件
img_path = "../dataset/train/bees/16838648_415acd9e3f.jpg"

# 使用 PIL 库的 Image.open() 函数打开指定路径的图像文件
# 该函数返回一个 PIL 图像对象，方便后续对图像进行操作
img_PIL = Image.open(img_path)

# 将 PIL 图像对象转换为 NumPy 数组
# # 因为 TensorBoard 更适合处理 NumPy 数组形式的数据，所以需要进行转换
img_array = np.array(img_PIL)
#
# # 可选择打印图像数组的形状，用于调试或查看图像的尺寸和通道信息
print(img_array.shape)

# 使用 writer 对象的 add_image 方法将图像数据添加到 TensorBoard 中
# "test_bees1" 是该图像在 TensorBoard 中的标签，方便识别和区分不同的图像
# 2 是全局步数，可用于表示图像在训练或其他过程中的顺序
# dataformats="HWC" 表示图像数组的数据格式为 高度（Height）、宽度（Width）、通道（Channel）
writer.add_image("test_bees1", img_array, 2, dataformats="HWC")

# 注释掉的代码，这里预留了添加标量数据的接口
# 可根据需求使用 writer.add_scalar() 方法添加各种标量数据，如损失值、准确率等
# writer.add_scalar()

# 以下注释掉的代码是一个示例，用于向 TensorBoard 中添加标量数据
# 这里模拟了一个简单的线性函数 y = 2x
# 通过 for 循环，从 0 到 99 生成 x 的值，计算对应的 y 值（2 * i）
# 并使用 writer.add_scalar() 方法将这些标量数据添加到 TensorBoard 中
# 标签为 "y=2x"，方便在 TensorBoard 中查看该数据的变化趋势
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

# 关闭 SummaryWriter 对象
# 确保所有缓存的数据都被写入日志文件，避免数据丢失
writer.close()
