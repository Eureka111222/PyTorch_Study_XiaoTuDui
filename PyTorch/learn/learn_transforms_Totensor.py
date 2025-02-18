#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 21:45
# @Author : Yuzhao Li

# 从 torchvision 库中导入 transforms 模块
# transforms 模块提供了一系列用于图像预处理的工具，例如图像的裁剪、缩放、归一化等操作
from torchvision import transforms
# 从 PIL 库中导入 Image 模块，用于打开和处理图像文件
from PIL import Image
# 导入 OpenCV 库，它是一个强大的计算机视觉库，可用于图像和视频处理
import cv2
# 从 torch.utils.tensorboard 模块中导入 SummaryWriter 类
# SummaryWriter 用于将数据写入 TensorBoard 所需的日志文件，方便后续可视化
from torch.utils.tensorboard import SummaryWriter

# 以下注释解释了 Python 中使用 transforms 模块将图像转换为 Tensor 数据类型的目的
# 在深度学习中，通常需要将图像数据转换为特定的数据类型（如 Tensor）才能输入到神经网络中进行处理
# 这里以 transforms.ToTensor() 为例，学习如何进行图像数据类型的转换以及为什么需要这种转换

# 定义要读取的图像文件的路径
# 可以使用绝对路径或相对路径，这里使用相对路径，假设当前脚本所在目录的上一级目录下有 dataset 文件夹
# 其中包含训练数据，ants 文件夹下有具体的图像文件
# 绝对路径示例：E:\DeskTop\deepl\pytorch\dataset\train\ants\0013035.jpg
# 相对路径示例：dataset/train/ants/0013035.jpg
#../表示当前目录的上一层目录  ./表示当前目录
img_path = "../dataset/train/ants/0013035.jpg"
# 使用 PIL 库的 Image.open() 函数打开指定路径的图像文件
# 该函数返回一个 PIL 图像对象，方便后续对图像进行操作
img = Image.open(img_path)
# 可选择打印 PIL 图像对象的信息，用于调试或查看图像的基本属性
# print(img)

# 1. transforms 的使用
# 创建 transforms.ToTensor() 类的实例
# transforms.ToTensor() 是一个用于将 PIL 图像或 NumPy 数组转换为 PyTorch Tensor 的变换操作
# 它会将图像的像素值归一化到 [0, 1] 范围，并调整维度顺序为 (C, H, W)，其中 C 是通道数，H 是高度，W 是宽度
tensor_trans = transforms.ToTensor()
# 调用 tensor_trans 实例，将 PIL 图像对象 img 转换为 Tensor 数据类型
tensor_img = tensor_trans(img)

# 打印转换后的 Tensor 图像数据，可用于查看数据的具体内容和格式
print(tensor_img)

# 创建一个 SummaryWriter 对象，指定日志文件的保存目录为 "../logs"
# 后续写入的数据（如图像、标量等）都会存储在该目录下的日志文件中
writer = SummaryWriter("../logs")
# 使用 writer 对象的 add_image 方法将 Tensor 图像数据添加到 TensorBoard 中
# "tensorimage" 是该图像在 TensorBoard 中的标签，方便识别和区分不同的图像
# 这里没有指定全局步数和数据格式，使用默认值
writer.add_image("tensorimage", tensor_img)
# 关闭 SummaryWriter 对象
# 确保所有缓存的数据都被写入日志文件，避免数据丢失
writer.close()

# 2. 为什么我们需要 Tensor 数据类型
# 在深度学习中，神经网络通常是使用 PyTorch 等深度学习框架构建的，这些框架中的模型和操作主要是针对 Tensor 数据类型设计的。
# Tensor 数据类型具有以下优点：
# - **高效计算**：Tensor 数据类型可以利用 GPU 进行加速计算，大大提高训练和推理的速度。
# - **自动求导**：PyTorch 中的 Tensor 支持自动求导机制，方便进行反向传播和梯度计算，这是训练神经网络的关键步骤。
# - **统一的数据格式**：Tensor 提供了统一的数据格式，使得不同的图像、文本等数据可以以相同的方式进行处理和输入到神经网络中。
# - **内置操作**：Tensor 数据类型提供了丰富的内置操作，如矩阵乘法、卷积等，方便实现各种神经网络层和算法。
# 因此，将图像数据转换为 Tensor 数据类型是深度学习中常见且必要的操作，以便能够充分利用深度学习框架的功能进行模型训练和推理。