#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 12:42
# @Author : Yuzhao Li
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

# 定义神经网络模型
class Yuzhao(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 使用 sequential
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# 加载图片
image_path = "../dataset/train/ants/28847243_e79fe052cd.jpg"
image = Image.open(image_path)
image = image.convert("RGB")

# 图片预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image)
# 增加一个维度，以符合模型输入的 batch 维度要求
image = image.unsqueeze(0)
print(image.shape)

# 加载预训练模型
model = torch.load("yuzhao_3.pth", map_location="cpu")
# 设置模型为评估模式
model.eval()

# 图片输入模型进行推理
with torch.no_grad():
    output = model(image)

# 处理预测结果
_, predicted = torch.max(output.data, 1)
# 假设 CIFAR-10 数据集的类别名称
classes = ('ant', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"预测结果: {classes[predicted.item()]}")