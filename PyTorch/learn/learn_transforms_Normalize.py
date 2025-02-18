#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 21:55
# @Author : Yuzhao Li

# 从 PIL 库中导入 Image 模块，用于打开和处理图像文件
from PIL import Image
# 从 torch.utils.tensorboard 模块中导入 SummaryWriter 类
# SummaryWriter 用于将数据写入 TensorBoard 所需的日志文件，方便后续可视化
from torch.utils.tensorboard import SummaryWriter
# 从 torchvision 库中导入 transforms 模块
# transforms 模块提供了一系列用于图像预处理的工具，例如图像的裁剪、缩放、归一化等操作
from torchvision import transforms

# 创建一个 SummaryWriter 对象，指定日志文件的保存目录为 "logs"
# 后续写入的数据（如图像、标量等）都会存储在该目录下的日志文件中
writer = SummaryWriter("../logs")

# 1. ToTensor 的使用
# 使用 PIL 库的 Image.open() 函数打开指定路径的图像文件
# 该函数返回一个 PIL 图像对象，方便后续对图像进行操作
img = Image.open("../dataset/train/ants/0013035.jpg")

# 创建 transforms.ToTensor() 类的实例
# transforms.ToTensor() 是一个用于将 PIL 图像或 NumPy 数组转换为 PyTorch Tensor 的变换操作
# 它会将图像的像素值归一化到 [0, 1] 范围，并调整维度顺序为 (C, H, W)，其中 C 是通道数，H 是高度，W 是宽度
trans_tensor = transforms.ToTensor()
# 调用 trans_tensor 实例，将 PIL 图像对象 img 转换为 Tensor 数据类型
tensor_img = trans_tensor(img)

# 使用 writer 对象的 add_image 方法将 Tensor 图像数据添加到 TensorBoard 中
# "Normalize_test" 是该图像在 TensorBoard 中的标签，方便识别和区分不同的图像
writer.add_image("Normalize_test_before", tensor_img)

# 2. Normalize 的使用
# 打印转换后的 Tensor 图像数据中第一个通道、第一行、第一列的像素值
# 用于查看转换后图像像素值的范围，因为 ToTensor 已经将像素值归一化到 [0, 1] 范围
print(tensor_img[0][0][0])

# 创建 transforms.Normalize() 类的实例
# transforms.Normalize() 用于对图像进行归一化操作，使图像数据具有零均值和单位方差
# 第一个参数 [0.5, 0.5, 0.5] 是每个通道的均值，第二个参数 [0.5, 0.5, 0.5] 是每个通道的标准差
# 归一化公式为：output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 调用 trans_norm 实例，对 Tensor 图像数据进行归一化操作
norm_img = trans_norm(tensor_img)

# 打印归一化后的 Tensor 图像数据中第一个通道、第一行、第一列的像素值
# 用于查看归一化后图像像素值的变化情况，归一化后像素值的范围通常会在 [-1, 1] 之间
print(norm_img[0][0][0])

# 使用 writer 对象的 add_image 方法将归一化后的 Tensor 图像数据添加到 TensorBoard 中
# "norm_sdq" 是该图像在 TensorBoard 中的标签，方便识别和区分不同的图像
writer.add_image("Normalize_test_after", norm_img)

# 关闭 SummaryWriter 对象
# 确保所有缓存的数据都被写入日志文件，避免数据丢失
writer.close()