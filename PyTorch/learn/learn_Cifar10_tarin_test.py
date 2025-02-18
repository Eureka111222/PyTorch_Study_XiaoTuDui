#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 12:14
# @Author : Yuzhao Li
import torch
import torchvision
from torch import argmax, nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 从 model 文件中导入相关内容，此处被注释，可能后续会使用
# from model import  *
import time

# 准备数据集
# 加载 CIFAR - 10 训练数据集
# "../testdataset" 为数据集存储路径
# train=True 表示加载训练集
# transform=torchvision.transforms.ToTensor() 将图像数据转换为 Tensor 类型
# download=True 若数据集不存在则进行下载
train_data = torchvision.datasets.CIFAR10("../testdataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 加载 CIFAR - 10 测试数据集
test_data = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 使用 len() 函数获取训练集和测试集的样本数量
train_data_size = len(train_data)
test_data_size = len(test_data)
# 格式化输出训练集和测试集的长度
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据集
# 使用 DataLoader 对训练数据集进行批量加载
# batch_size=64 表示每个批次包含 64 个样本
train_dataloader = DataLoader(train_data, batch_size=64)
# 对测试数据集进行批量加载
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Yuzhao(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)
        # 使用 nn.Sequential 按顺序组合多个神经网络层
        self.model1 = nn.Sequential(
            # 第一个卷积层，输入通道 3，输出通道 32，卷积核大小 5，填充 2
            Conv2d(3, 32, 5, padding=2),
            # 第一个最大池化层，池化核大小 2
            MaxPool2d(2),
            # 第二个卷积层，输入通道 32，输出通道 32，卷积核大小 5，填充 2
            Conv2d(32, 32, 5, padding=2),
            # 第二个最大池化层，池化核大小 2
            MaxPool2d(2),
            # 第三个卷积层，输入通道 32，输出通道 64，卷积核大小 5，填充 2
            Conv2d(32, 64, 5, padding=2),
            # 第三个最大池化层，池化核大小 2
            MaxPool2d(2),
            # 展平层，将多维张量展平为一维
            Flatten(),
            # 第一个全连接层，输入特征 1024，输出特征 64
            Linear(1024, 64),
            # 第二个全连接层，输入特征 64，输出特征 10
            Linear(64, 10)
        )

    def forward(self, x):
        # 定义前向传播过程，将输入数据通过 model1 进行处理
        x = self.model1(x)
        return x

# 创建 Yuzhao 类的实例
yuzhao = Yuzhao()

# 检查是否有可用的 CUDA 设备，如果有则将模型移动到 GPU 上
if torch.cuda.is_available():
    yuzhao = yuzhao.cuda()

# 定义损失函数，使用交叉熵损失函数，常用于分类任务
loss_fn = nn.CrossEntropyLoss()
# 如果有可用的 CUDA 设备，将损失函数移动到 GPU 上
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 定义优化器的学习率
learning_rate = 1e-2
# 使用随机梯度下降（SGD）优化器，对模型的参数进行优化
optimizer = torch.optim.SGD(yuzhao.parameters(), lr=learning_rate)

# 设置网络训练的一些参数
# 记录训练的总步数
total_train_step = 0
# 记录测试的总步数
total_test_step = 0
# 定义训练的总轮数
epoch = 10

# 创建 SummaryWriter 对象，用于将训练和测试过程中的信息写入 TensorBoard 日志
writer = SummaryWriter("../logs")

# 开始训练过程
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))

    # 记录训练开始时间
    start_time = time.time()
    # yuzhao.train()，用于将模型设置为训练模式
    # 此模式对部分特殊层（如 Dropout 等）有影响
    # yuzhao.train()

    # 遍历训练数据加载器中的每个批次
    for data in train_dataloader:
        # 解包每个批次的数据，得到图像数据和对应的标签
        imgs, targets = data
        # 如果有可用的 CUDA 设备，将图像数据和标签移动到 GPU 上
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        # 将图像数据输入到模型中进行前向传播，得到预测输出
        outputs = yuzhao(imgs)
        # 计算预测输出和真实标签之间的损失
        loss = loss_fn(outputs, targets)

        # 优化器进行参数更新的步骤
        # 清空优化器中之前计算的梯度
        optimizer.zero_grad()
        # 进行反向传播，计算损失函数关于网络参数的梯度
        loss.backward()
        # 根据计算得到的梯度更新网络参数
        optimizer.step()

        # 训练步数加 1
        total_train_step += 1
        # 记录当前时间
        end_time = time.time()
        # 每训练 100 步，打印训练信息并写入 TensorBoard 日志
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
            print("用时:{}".format(end_time - start_time))
            # 将训练损失写入 TensorBoard 日志
            writer.add_scalar("train_loss", loss, total_train_step)

    # 使用测试集测试模型
    # yuzhao.eval()，用于将模型设置为评估模式
    # 此模式对部分特殊层（如 Dropout 等）有影响
    yuzhao.eval()

    # 初始化测试集的总损失和总正确预测数
    total_test_loss = 0
    total_acc_num = 0
    # 不计算梯度，节省计算资源
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次
        for data in test_dataloader:
            # 解包每个批次的数据，得到图像数据和对应的标签
            imgs, targets = data
            # 如果有可用的 CUDA 设备，将图像数据和标签移动到 GPU 上
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            # 将图像数据输入到模型中进行前向传播，得到预测输出
            outputs = yuzhao(imgs)
            # 计算预测输出和真实标签之间的损失
            loss = loss_fn(outputs, targets)

            # 计算当前批次的正确预测数
            acc_num = (outputs.argmax(1) == targets).sum()
            # 累加总正确预测数
            total_acc_num += acc_num
            # 累加总测试损失
            total_test_loss += loss

        # 计算测试集的准确率
        total_accuracy = total_acc_num / test_data_size
        print("整体测试集上的 Loss:{}".format(total_test_loss))
        print("整体测试集上的正确率:{}".format(total_accuracy))
        # 将测试损失写入 TensorBoard 日志
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        # 将测试准确率写入 TensorBoard 日志
        writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
        # 测试步数加 1
        total_test_step += 1

    # 保存模型
    # 注释掉的代码，将整个模型保存到文件中，文件名包含当前轮数
    torch.save(yuzhao,"yuzhao_{}.pth".format(i+1))

# 关闭 SummaryWriter 对象，确保所有数据都写入日志文件
writer.close()
