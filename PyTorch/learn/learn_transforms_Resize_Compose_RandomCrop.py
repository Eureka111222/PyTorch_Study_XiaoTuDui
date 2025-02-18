#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 22:06
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
trans_totensor = transforms.ToTensor()
# 调用 trans_totensor 实例，将 PIL 图像对象 img 转换为 Tensor 数据类型
tensor_img = trans_totensor(img)

# 使用 writer 对象的 add_image 方法将 Tensor 图像数据添加到 TensorBoard 中
# "Origin_ant" 是该图像在 TensorBoard 中的标签，方便识别和区分不同的图像
writer.add_image("Origin_ant", tensor_img)

# 2. Normalize 的使用
# 打印转换后的 Tensor 图像数据中第一个通道、第一行、第一列的像素值
# 用于查看转换后图像像素值的范围，因为 ToTensor 已经将像素值归一化到 [0, 1] 范围
print("未归一化时第一个元素:")
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
print("归一化后第一个元素:")
print(norm_img[0][0][0])

# 使用 writer 对象的 add_image 方法将归一化后的 Tensor 图像数据添加到 TensorBoard 中
# "Norm_ant" 是该图像在 TensorBoard 中的标签，方便识别和区分不同的图像
writer.add_image("Norm_ant", norm_img)

# 3. Resize 的使用
# 打印原始 PIL 图像的尺寸，格式为 (宽度, 高度)
print("原始图像的尺寸:")
print(img.size)

# 创建 transforms.Resize() 类的实例，指定目标尺寸为 (512, 200)
# 该操作会将图像的宽度调整为 512，高度调整为 200
trans_resize = transforms.Resize((512, 200))
# 调用 trans_resize 实例，对 PIL 图像对象 img 进行尺寸调整
resize_img = trans_resize(img)

# 打印调整尺寸后的 PIL 图像的尺寸，格式为 (宽度, 高度)
print("Resize（变形）后图像的尺寸:")
print(resize_img.size)

# 将调整尺寸后的 PIL 图像转换为 Tensor 数据类型
img_reszietensor = trans_totensor(resize_img)
# 使用 writer 对象的 add_image 方法将调整尺寸后的 Tensor 图像数据添加到 TensorBoard 中
# "resize" 是该图像在 TensorBoard 中的标签，0 是全局步数，用于标识图像的顺序
writer.add_image("Resize512*200_ant", img_reszietensor, 0)

# 4. Compose 的使用（类似于一个流水线，里面的参数是一道道的工序）
# 创建 transforms.Resize() 类的实例，只指定一个参数 333
# 此时图像会按比例缩放，使得较短的边长度为 333
trans_resize_2 = transforms.Resize(333)

# 创建 transforms.Compose() 类的实例，将多个变换操作组合在一起
# 这里将 trans_resize_2 和 trans_totensor 组合成一个流水线，先进行尺寸调整，再转换为 Tensor 类型
# 输入的参数类型一定要符合流水线各个工序的要求，此例输入 PIL 图像 -> 调整尺寸后的 PIL 图像 -> Tensor 图像
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])

# 调用 trans_compose 实例，对 PIL 图像对象 img 进行处理
img_resize_2 = trans_compose(img)

# 使用 writer 对象的 add_image 方法将处理后的 Tensor 图像数据添加到 TensorBoard 中
# "resize" 是该图像在 TensorBoard 中的标签，2 是全局步数，用于标识图像的顺序
writer.add_image("Resize333_ant", img_resize_2, 2)

# 5. RandomCrop: 随机裁剪的使用
# 创建 transforms.RandomCrop() 类的实例，指定裁剪尺寸为 (100, 200)
# 该操作会在图像上随机选择一个区域进行裁剪，裁剪后的图像尺寸为 100x200
# 也可以只指定一个参数，如 trans_randomcrop = transforms.RandomCrop(100)，此时裁剪区域为正方形
trans_randomcrop = transforms.RandomCrop((100, 200))

# 创建 transforms.Compose() 类的实例，将随机裁剪和转换为 Tensor 类型的操作组合在一起
# compose的使用（类似于一个流水线，里面的参数是一道道的工序）
# 输入的参数类型一定要符合流水线各个工序的要求，此例输入PIL->PIL->tensor
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_totensor])

# 循环 10 次，每次对原始 PIL 图像进行随机裁剪并转换为 Tensor 类型
# 然后将处理后的 Tensor 图像数据添加到 TensorBoard 中，使用不同的全局步数 i 来区分不同的裁剪结果
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("randomcropHW", img_crop, i)

# 关闭 SummaryWriter 对象
# 确保所有缓存的数据都被写入日志文件，避免数据丢失
writer.close()