

&nbsp;&nbsp;&nbsp;&nbsp;近期科研任务加重，看了许多文章，发现大部分论文项目都是自己手搓的（或多或少，有的是在某一个手搓论文项目上改的），而自己之前学过的PyTorch基础又忘光了，看起代码费力的我哇哇叫，决定重新复习一下PyTorch，并做一下笔记，便于自己复习，也供大家一起学习，==快速入门PyTorch!!==

</br>
</br>

<font color='red' size='5'>**参考课程：**[b站的小土堆PyTorch快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=58c0d5d18626a7a9c1f33901c6769f56)</font>
<font color='red'  size='5'>**参考教程：**[ PyTorch 教程](https://pytorch.cadn.net.cn/)</font>
<font color='red'  size='5'>**项目源码：**[ github项目源码点击下载](https://github.com/Eureka111222/PyTorch_Study_XiaoTuDui.git/)</font>

# PyTorch学习
@[TOC](目录)
# 1. PyTorch环境搭建
&nbsp;&nbsp;&nbsp;&nbsp;因环境第一次学习在本地Windows和实验室上已经安好，这里不再重复，因为==安装配置环境也挺重要、细节蛮多的，建议大家跟着专门环境安装博客进行环境安装==，这里主要记录使用学习。
建议参考：[PyTorch官网](https://pytorch.org/)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2024最新Pytorch安装教程（同时讲解安装CPU和GPU版本）](https://blog.csdn.net/Little_Carter/article/details/135934842?fromshare=blogdetail&sharetype=blogdetail&sharerId=135934842&sharerefer=PC&sharesource=m0_52181935&sharefrom=from_link)
==安装GPU版本一定注意这！！==
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d920317e1924765a873320c04a06e2e.png)
==**还有版本对应关系，很早直接保存的（实在忘记那里的了，抱歉没引用），因为自己创建环境，经常因为版本不对应直接报大错**==
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9b425743b17648e8ac0c96a086d77c28.png)
我的有关版本
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/756422b88e774fb4a90984e1acc22b3b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1347343011f74e2fb948f7d00936e816.png)


```python
conda activate pytorch 激活pytorch
conda env list 查看已经创建的环境
pip list  查看已经安装的库
终端输入python：进入python   quit():退出python
torch.cuda.is_available() 查看pytorch是否可以用gpu（在python下用）
换成conda install -n pytorch ipykernel （或者其他安装jupyter）
dir():打开，查看里面有什么
help():说明文档

import torch
 
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  #输出为True，则安装成功
dir(torch)
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f4f8225f18364e2e9768e868179e2153.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bed9122d0e824102aef6cc7a2a43c422.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b3e5075b4c843a49d0533637646be23.png)


Python三种运行方式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f4d2745286d5492997550b8c672fa5e9.png)
</br>
</br>
</br>
==<font color='red' size='5'>**以下所有代码都在Windows10下的PyCharm中以Python文件运行**</font>==

# 2. 数据加载(Dataset)

>常见的三种数据集形式：（1）文件夹名是分类（label），里面是图片
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（2）图片和label分开放在两个不同的文件夹下，图片分别对应不同的分类（比如存在txt文件下）
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;（3）图片的命名就是分类
> ==以下三张图片对应三种形式==

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3270b6326f57418b9f9f8c9334b839a6.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae655e4553bc4987b0e87552469617d5.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b268741f6b1d477089656d80831dd5be.png)
> Dataset用于构建数据集

- learn_read_data.py

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 21:05
# @Author : Yuzhao Li

from torch.utils.data import  Dataset
from PIL import Image
import cv2
import os

#MyData类集成自Dataset类
class MyData(Dataset):
    #将输入的路径拼接并转成一个list
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir  #获取根目录
        self.label_dir=label_dir #子目录
        self.path=os.path.join(self.root_dir,self.label_dir) #将根目录与子目录拼接
        self.img_path=os.listdir(self.path) #将这个文件夹下的东西变成一个list

    #获取具体的一个数据->在训练时一般有这个函数
    def __getitem__(self, idx):
        img_name=self.img_path[idx] #获取图片名称
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    #获取数据集长度
    def __len__(self):
        return len(self.img_path)

#测试
root_dir= "../dataset/train"
ants_label="ants"
bees_label="bees"
ants_dataset=MyData(root_dir,ants_label) #构建具体的数据集
bees_dataset=MyData(root_dir,bees_label)

# #展示第一张蚂蚁数据集的图片
# img,label=ants_dataset[0]
# print(label)
# img.show()

# 将两个数据集合并为一个
train_dataset=ants_dataset+bees_dataset
print(len(train_dataset))

# 展示数据集第125张图像，因为ants只有124张图片，所以第125张图片就是bees的第一张，是一个蜜蜂
img,label=train_dataset[124]
print(label)
img.show()
```

# 3.日志（torch.utils.tensorboard）

> 训练时一些训练信息（比如损失函数等）可以用tensorboard展示，其会生成一个.log文件，用浏览器打开
> 指令：tensorboard --logdir=logs 
> ==logs即存放日志的目录，不用详细的写哪一个日志== ，打开tensorboard的logs，主要是scalar和image的使用
>

- learn_tensorboard.py
>使用 PyTorch 的 tensorboard 工具将一张图片和一个简单的标量数据（这里模拟 y = 2x）记录下来，以便后续使用 TensorBoard 进行可视化展示。
```python
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

```
- 打开tensorboard的logs实例
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d1cac96548914296acd5f57b67b762e5.png)


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d276e3f4dd7e495da9b44fb439803df6.png)
# 4. torchvision.tensorforms
>torchvision.tensorforms工具箱包含了一些对图像进行Tensor处理的基本流程，不同于计算机的数组，在进行深度学习图像处理时，会使用一种类似数组的tensor（张量的形式），便于计算
- 将图片作为输入，输入到tensorforms中，经过其中一系列操作（就是调用其中的函数等）转成输出，下面第一幅是粗略流程，第二幅中展示了调用其中函数的细节
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/712419686c8a4edd81f5385d916223fa.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1bd56fb8a79b4aa4855cd84a1c6484bb.png)


## 4.1 ToTensor
- learn_transforms_Totensor.py
```python
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
```
 - 转成的tensor形式
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b600bfda6a9642558be35cdc6745fe50.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca2200532e844218bbad57b2e06ed1cd.png)

## 4.2 Normalize
- learn_transforms_Normalize.py
```python
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
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0247e361aeea4821b463e95bcb2753fd.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4c3ffbcc955e490196860623fadd99dc.png)
## 4.3 Resize、Compose、RandomCrop

learn_transforms_Resize_Compose_RandomCrop.py

```python
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
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/32463f2c883246eaa39c2c26728ff735.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9295bcf398fa43ee9f3c0ed618e3f51a.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/89bdf21de3544bb88b65d2eae6fed229.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/657f5cc9e65943c199607f2f25903c4d.png)
# 5. transforms+torchvision.datasets联合使用
>torchvision.datasets 是 PyTorch 中 torchvision 库的一个重要模块，它提供了丰富的计算机视觉常用数据集的接口，方便用户快速加载和使用这些数据集进行模型的训练、评估等工作。
- learn_transforms.datasets.py
```python
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
    # 创建 SummaryWriter 对象，将日志文件保存到 logs目录下
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
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13b17f8889154b96881938cc0ef83599.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be0d125ed7244a62b3fdc0f3d5c5a361.png)

# 6. transforms+torchvision.datasets+Dataloader联合使用
>DataLoader：创建一个数据加载器，将测试数据集按照 batch_siz 进行批量加载，shuffle表示是否打乱数据顺序，num_workers表示是否使用多线程加载数据，drop_last 表示是否丢弃最后一个不完整的批次（因为有有时最后一些图片不足以一个epoch）。

- learn_Dataloader.py

```python
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

```


# 7. nn.Module
>nn.Module 是 PyTorch 中所有神经网络模块的基类，它提供了以下重要功能：
>1. 参数管理
当你在 nn.Module 的子类中定义 nn.Parameter 类型的属性时，nn.Module 会自动将这些参数注册到模型中。例如，一个全连接层 nn.Linear 内部的权重和偏置就是 nn.Parameter 类型，它们会被自动管理。
可以使用==model.parameters() 方法获取模型的所有可训练参数==，这在定义优化器时非常有用。
>2. 模块嵌套和注册
可以==在 nn.Module 的子类中嵌套其他 nn.Module 实例==，nn.Module 会自动管理这些子模块。例如，一个复杂的神经网络可能由多个全连接层、卷积层等子模块组成。
子模块会被自动注册到父模块中，可以使用 model.children() 或 model.modules() 方法遍历子模块。
>3. 自动求导支持
nn.Module 与 PyTorch 的自动求导机制紧密结合。当你定义了前向传播过程（forward 方法）后，==PyTorch 会自动根据计算图计算梯度==，方便进行反向传播和参数更新。
>4. 模型保存和加载
由于 nn.Module 管理着模型的参数和结构，因此==可以方便地使用 torch.save() 和 torch.load() 方法保存和加载模型。==
>5. 前向传播定义
==自定义的 nn.Module 子类必须实现 forward 方法==，该方法定义了数据从输入到输出的计算过程。在调用模型实例时，实际上会调用 forward 方法进行前向传播计算。

- learn_nn.module.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 9:35
# @Author : Yuzhao Li

import torch
from torch import nn

# 创建自己的网络类，继承自 nn.Module
# nn.Module 是 PyTorch 中所有神经网络模块的基类，自定义的神经网络模型通常需要继承这个类
# 通过继承 nn.Module，我们可以方便地利用 PyTorch 提供的自动求导、参数管理等功能
class Yuzhao(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        # 调用父类 nn.Module 的构造函数，确保父类的初始化逻辑被正确执行
        # 这样可以继承父类的属性和方法，如参数管理、模块注册等
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # 定义前向传播过程，这是自定义神经网络模型必须实现的方法
        # 前向传播是指数据从输入层经过一系列的计算和变换得到输出的过程
        # 在这个简单的例子中，我们对输入进行了一个简单的加 1 操作
        output = input + 1
        return output

# 创建 Yuzhao 类的实例
yuzhao = Yuzhao()

# 创建一个包含单个元素 1.0 的张量作为输入
x = torch.tensor(1.0)

# 调用 yuzhao 实例，实际上会调用其 forward 方法进行前向传播计算
# 在 PyTorch 中，调用一个继承自 nn.Module 的实例时，会自动调用其 forward 方法
output = yuzhao(x)

# 打印前向传播的输出结果
print(output)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/34619a593ae54502a02f99363c752534.png)

# 8. 二维卷积（torch.nn.functional.conv2d）
>使用torch.nn.functional下的conv2d实现二维卷积
- learn_conv2d.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 9:43
# @Author : Yuzhao Li

import torch
import torch.nn.functional as F

# 定义一个 2 维的输入张量
# 这里模拟了一个 5x5 的图像矩阵，每个元素代表一个像素值
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 定义卷积核（算子）
# 卷积核是一个 3x3 的矩阵，用于在输入张量上进行卷积操作
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 改变输入张量和卷积核的形状
# 在 PyTorch 中，卷积层的输入和卷积核需要满足特定的形状要求
# 对于 2D 卷积，输入的形状应该是 (batch_size, in_channels, height, width)
# 这里 batch_size 为 1，表示只有一个样本；in_channels 为 1，表示输入只有一个通道
# 卷积核的形状应该是 (out_channels, in_channels, kernel_height, kernel_width)
# 这里 out_channels 为 1，表示输出只有一个通道；in_channels 为 1，表示输入只有一个通道
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# 打印输入张量和卷积核的形状，方便检查是否符合要求
print(input.shape)
print(kernel.shape)

# 使用 torch.nn.functional.conv2d 函数进行 2D 卷积操作
# input 是输入张量
# kernel 是卷积核
# stride=1 表示卷积核在输入张量上滑动的步长为 1
# 该函数会返回卷积操作后的输出张量
output = F.conv2d(input, kernel, stride=1)

# 打印卷积操作的输出结果
print(output)
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9e61c54c317a4df9b1238d3b255540f9.png)

# 9. 二维卷积（nn.conv2d）
- learn_nn.conv2d.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 9:50
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
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
            self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

        def forward(self, x):
            x = self.conv1(x)
            return x

    # 创建模型实例
    yuzhao = Yuzhao()
    # 创建 TensorBoard 写入器
    writer = SummaryWriter("../logs")

    step = 0
    try:
        # 遍历数据加载器中的每个批次数据
        for data in dataloader:
            imgs, targets = data
            # 将图像数据输入到模型中进行卷积操作
            output = yuzhao(imgs)
            # 写入输入图像到 TensorBoard
            writer.add_images("input", imgs, step)

            # 分别可视化卷积层的 6 个输出通道
            for channel in range(output.shape[1]):
                single_channel_output = output[:, channel:channel + 1, :, :]
                writer.add_images(f"output_channel_{channel}", single_channel_output, step)

            step += 1
            print(imgs.shape)
            print(output.shape)
    except Exception as e:
        print(f"模型运行出错: {e}")
    finally:
        # 关闭 TensorBoard 写入器
        writer.close()
        print("end")
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/361f645fb6214fbb9683725b6b1234c1.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf257d72e6d34040b0d13dddb6621027.png)
>在 PyTorch 中，nn.Conv2d 和 torch.nn.functional.conv2d 都用于实现二维卷积操作，但它们在使用方式和应用场景上存在一些区别
>1.nn.Conv2d 是 torch.nn 模块中的一个类，用于创建一个二维卷积层对象。它封装了卷积操作所需的参数（如卷积核的权重和偏置），并提供了一种面向对象的方式来构建神经网络。
>```python
>torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
>```
>in_channels：输入特征图的通道数，例如对于彩色图像，通常为 3（RGB 三个通道）。
out_channels：输出特征图的通道数，即卷积核的数量。
kernel_size：卷积核的大小，可以是一个整数（表示正方形卷积核）或一个元组（表示不同的高度和宽度）。
stride：卷积核在输入特征图上滑动的步长，默认为 1。
padding：在输入特征图的边缘填充的像素数，默认为 0。
dilation：卷积核元素之间的间距，默认为 1。
groups：输入通道和输出通道之间的分组连接方式，默认为 1。
bias：是否添加偏置项，默认为 True。
padding_mode：填充模式，默认为 'zeros'，即使用零填充。
>2.torch.nn.functional.conv2d 是一个函数，用于直接执行二维卷积操作。它不封装卷积层的参数，需要手动传入卷积核的权重和偏置（如果需要）。
>```python
>torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
>```
>input：输入的特征图张量，形状为 (batch_size, in_channels, height, width)。
weight：卷积核的权重张量，形状为 (out_channels, in_channels // groups, kernel_height, kernel_width)。
bias：可选的偏置项张量，形状为 (out_channels,)。
stride、padding、dilation、groups：与 nn.Conv2d 中的参数含义相同。
- 两者的区别
1. 面向对象 vs 函数式
nn.Conv2d 是一个类，属于面向对象的编程方式。使用时需要先创建卷积层对象，然后将输入数据传递给该对象进行卷积操作。它会自动管理卷积核的权重和偏置等参数，并且可以方便地集成到 nn.Module 子类中构建复杂的神经网络。
torch.nn.functional.conv2d 是一个函数，属于函数式编程方式。使用时需要手动传入卷积核的权重和偏置，更适合在需要灵活控制卷积操作，或者不希望封装参数的场景中使用。
2. 参数管理
nn.Conv2d 会自动管理卷积层的参数，例如可以通过 conv_layer.parameters() 方法获取卷积层的所有可训练参数，方便在优化器中使用。
torch.nn.functional.conv2d 不管理参数，需要手动定义和更新卷积核的权重和偏置。
3. 模型构建和复用
nn.Conv2d 更适合用于构建复杂的神经网络模型，因为它可以方便地与其他 nn.Module 子类组合使用，并且可以在不同的地方复用同一个卷积层对象。
torch.nn.functional.conv2d 更适合在一些简单的脚本或需要自定义卷积操作逻辑的场景中使用。
4. 状态管理
nn.Conv2d 可以保存和加载卷积层的状态（权重和偏置），方便模型的保存和恢复。
torch.nn.functional.conv2d 不涉及状态管理，每次调用都需要手动传入参数。

#  10. 池化操作（nn.MaxPool）
>在 PyTorch 中，nn.MaxPool 相关的类主要用于实现最大池化（Max Pooling）操作，最大池化是卷积神经网络（CNN）中常用的一种下采样技术，它能够在保留特征图中重要特征的同时，减少数据的维度，从而降低计算量和模型的过拟合风险。最大池化操作会在输入数据上滑动一个固定大小的池化窗口，在每个窗口内选择最大值作为输出。例如，对于一个 2x2 的池化窗口，它会在输入数据上每次移动一定的步长，将窗口内的 4 个元素中的最大值作为输出的一个元素。
>```python
>torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
>torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
>torch.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
>```
>kernel_size：池化窗口的大小，可以是一个整数。
>stride：池化窗口移动的步长，默认为 kernel_size。
>padding：在输入数据的边缘填充的大小，默认为 0。
>dilation：控制池化窗口内元素之间的间距，默认为 1。
>return_indices：是否返回最大值的索引，默认为 False。
>ceil_mode：当池化窗口超出输入边界时，是否使用 ceil 函数计算输出大小，默认为 False。
>
- learn_maxpool.py
```python
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
    writer = SummaryWriter("../logs/maxpool")

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
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb68b9ba47cf4eecaf8ee98a2e88ea19.png)


# 11.  非线性激活函数（Sigmoid、Relu）
>用于增加非线性
- learn_SigRelu.py
```python
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
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fa8f65565b4e40c39bd19c14685f2189.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/29b1c61e90e143109962f1573b06b05f.png)

# 12. 线性层(Linear)
> nn.Linear 是 PyTorch 中用于定义全连接层（也称为线性层）的类，在神经网络中应用广泛，常用于将输入特征进行线性变换。
> 原理：
全连接层的本质是一个线性变换，对于输入向量x ，通过一个权重矩阵 W和一个偏置向量 b进行如下计算得到输出向量 ：
y=Wx+b
其中， W是一个形状为 (out_features, in_features) 的矩阵，b是一个形状为 (out_features,) 的向量。
>```python
>torch.nn.Linear(in_features, out_features, bias=True)
>```
>in_features：输入特征的数量，也就是输入向量的维度。
out_features：输出特征的数量，也就是输出向量的维度。
bias：一个布尔值，默认为 True，表示是否使用偏置项。如果为 True，则会在计算中加上偏置向量 ；如果为 False，则不使用偏置项。

- learn_Linear.py

 
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:15
# @Author : Yuzhao Li
import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载 CIFAR - 10 测试数据集
# root 参数指定数据集存储的路径为 "../testdataset"
# train=False 表示加载的是测试集
# transform=torchvision.transforms.ToTensor() 表示将图像数据转换为 Tensor 类型
# download=True 表示如果数据集不存在则进行下载
dataset = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 创建 DataLoader 对象，用于批量加载数据集
# batch_size=64 表示每个批次包含 64 个样本
dataloader = DataLoader(dataset, batch_size=64)

# 定义自定义的神经网络类，继承自 nn.Module
class Yuzhao(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)
        # 定义一个全连接层（线性层）
        # 输入特征的数量为 196608
        # 输出特征的数量为 10
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        # 定义前向传播过程
        # 将输入数据通过线性层进行变换
        output = self.linear1(input)
        return output

# 创建自定义神经网络的实例
yuzhao = Yuzhao()

# 遍历数据加载器中的每个批次的数据
for data in dataloader:
    # 解包每个批次的数据，得到图像数据和对应的标签
    imgs, targets = data
    # 打印图像数据的形状
    print("图像原始维度：")
    print(imgs.shape)
    # 尝试将图像数据展平，但此处代码只是注释掉的调用，未实际执行
    # torch.flatten(imgs)
    # 将图像数据进行形状重塑
    # 这里使用 -1 让 PyTorch 自动计算该维度的大小
    # 最终将数据重塑为形状 (1, 1, 1, -1)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    # 打印重塑后数据的形状
    # print(output.shape)
    # 将重塑后的数据输入到自定义神经网络中进行前向传播
    output = yuzhao(output)
    print("图像经过Linear后维度")
    # 打印经过神经网络处理后输出数据的形状
    print(output.shape)
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/47806086a6a0440186235dce25577f78.png)

# 13. 序列层（Sequential）
>==Sequential== 是 PyTorch 中的一个容器类，用于按顺序组织多个神经网络层。它允许你将一系列的网络层组合在一起，形成一个连续的计算模块。使用 Sequential 的好处是可以简化代码结构，避免在 forward 方法中逐个调用每个网络层。
> ==Flatten==是 PyTorch 中的一个网络层，用于将多维的输入张量展平为一维张量。在卷积神经网络中，卷积层和池化层的输出通常是多维的张量，而全连接层要求输入是一维的向量，因此需要使用 Flatten 层将多维张量展平。
>原理:Flatten 层会将输入张量的除了第一个维度（通常是批量大小）之外的所有维度进行合并，将其转换为一维向量。
- learn_sequential.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:22
# @Author : Yuzhao Li
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


# 定义自定义的神经网络类 Yuzhao，继承自 nn.Module
class Yuzhao(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        # 调用父类的构造函数，确保父类的初始化逻辑被执行
        super().__init__(*args, **kwargs)

        # 原本逐个定义网络层的方式，被注释掉了
        # self.conv1=Conv2d(3,32,5,padding=2)  # 第一个卷积层，输入通道 3，输出通道 32，卷积核大小 5，填充 2
        # self.maxpool1=MaxPool2d(2)  # 第一个最大池化层，池化核大小 2
        # self.conv2=Conv2d(32,32,5,padding=2)  # 第二个卷积层，输入通道 32，输出通道 32，卷积核大小 5，填充 2
        # self.maxpool2=MaxPool2d(2)  # 第二个最大池化层，池化核大小 2
        # self.conv3=Conv2d(32,64,5,padding=2)  # 第三个卷积层，输入通道 32，输出通道 64，卷积核大小 5，填充 2
        # self.maxpool3=MaxPool2d(2)  # 第三个最大池化层，池化核大小 2
        # self.flatten=Flatten()  # 用于将多维张量展平为一维
        # self.linear1=Linear(1024,64)  # 第一个全连接层，输入特征 1024，输出特征 64
        # self.linear2=Linear(64,10)  # 第二个全连接层，输入特征 64，输出特征 10

        # 使用 Sequential 容器来组织网络层
        # Sequential 可以按顺序依次执行其中的网络层，简化代码结构
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 第一个卷积层
            MaxPool2d(2),  # 第一个最大池化层
            Conv2d(32, 32, 5, padding=2),  # 第二个卷积层
            MaxPool2d(2),  # 第二个最大池化层
            Conv2d(32, 64, 5, padding=2),  # 第三个卷积层
            MaxPool2d(2),  # 第三个最大池化层
            Flatten(),  # 展平层
            Linear(1024, 64),  # 第一个全连接层
            Linear(64, 10)  # 第二个全连接层
        )

    def forward(self, x):
        # 原本逐个调用网络层的前向传播方式，被注释掉了
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)

        # 直接通过 Sequential 容器进行前向传播
        x = self.model1(x)

        return x

# 创建 Yuzhao 类的实例
yuzhao = Yuzhao()
# 打印模型的结构信息
print(yuzhao)

# 构造一个测试输入数据
# 形状为 (64, 3, 32, 32)，表示批量大小为 64，通道数为 3，高度和宽度均为 32 的输入张量
input = torch.ones(64, 3, 32, 32)
# 将输入数据传入模型进行前向传播
output = yuzhao(input)
# 打印输出数据的形状
print(output.shape)

# 创建一个 SummaryWriter 对象，用于将模型结构写入 TensorBoard 日志
writer = SummaryWriter("../logs")
# 将模型的计算图添加到 TensorBoard 中，方便可视化模型结构
writer.add_graph(yuzhao, input)
# 关闭 SummaryWriter 对象，确保所有数据都被写入日志文件
writer.close()
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e5f2655f10904f83a85f6b6d79cc5714.png)



# 14. 损失函数与反向传播（Loss）
## 14.1 各种损失函数
![>](https://i-blog.csdnimg.cn/direct/65ab8f71d8cf47b6a5affb4f36a129d3.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a95078ad91e444a0994f81c5afab6792.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7873cad4abf84f51b190dd30d54e0fa3.png)


- learn_Loss.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:30
# @Author : Yuzhao Li
import torch
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

# 定义输入张量，模拟模型的预测输出
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# 定义目标张量，即真实的标签值
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# 调整输入张量的形状，使其符合四维张量的格式 (batch_size, channels, height, width)
# 这里 batch_size 为 1，channels 为 1，height 为 1，width 为 3
inputs = torch.reshape(inputs, (1, 1, 1, 3))
# 同样调整目标张量的形状
targets = torch.reshape(targets, (1, 1, 1, 3))

# 创建 L1 损失函数对象
# L1 损失（也称为平均绝对误差，MAE）是预测值与真实值之间绝对误差的平均值
# 默认情况下，reduction 参数为 'mean'，表示取平均值
# 这里将 reduction 参数设置为 'sum'，表示对所有误差求和
# loss = L1Loss()  # 默认取平均值
loss = L1Loss(reduction="sum")
# 计算输入和目标之间的 L1 损失
result = loss(inputs, targets)
# 打印 L1 损失的计算结果
print(result)

# 创建均方误差（MSE）损失函数对象
# MSE 损失是预测值与真实值之间误差平方的平均值
loss_mse = MSELoss()
# 计算输入和目标之间的 MSE 损失
result_mse = loss_mse(inputs, targets)
# 打印 MSE 损失的计算结果
print(result_mse)

# 定义一个新的输入张量，模拟模型的预测输出（通常是未经过 softmax 处理的 logits）
x = torch.tensor([0.1, 0.2, 0.3])
# 定义对应的目标标签，这里表示真实类别为 1
y = torch.tensor([1])
# 调整输入张量的形状，使其符合二维张量的格式 (batch_size, num_classes)
# 这里 batch_size 为 1，num_classes 为 3
x = torch.reshape(x, (1, 3))
# 创建交叉熵损失函数对象
# 交叉熵损失常用于分类问题，它衡量的是两个概率分布之间的差异
loss_cross = CrossEntropyLoss()
# 计算输入和目标之间的交叉熵损失
# 交叉熵损失的计算公式为：loss = -x[class] + log(求和(exp(xi)))
result_cross = loss_cross(x, y)
# 打印交叉熵损失的计算结果
print(result_cross)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e21f6bce2eb43babbf1f9f6e36915a1.png)


## 14.2 损失函数结合CIFAR10
下面代码因为没有使用优化器，就只是计算每个batch_size的损失
- learn_Loss_Cifar10.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:39
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

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
            self.model1 = Sequential(
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

    # 创建模型实例
    yuzhao = Yuzhao()
    # 创建交叉熵损失函数
    loss_cross = nn.CrossEntropyLoss()
    
    for data in dataloader:
        imgs, targets = data
        outputs = yuzhao(imgs)
        # print(outputs)
        # print(targets)
        # 使用loss函数
        results = loss_cross(outputs, targets)
        # 反向传播计算梯度
        grad = results.backward()
        print(results)

```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6bf7a8a305ca4cb98f9700733b9caf2b.png)

# 15. 优化器
>在深度学习中，优化器（Optimizer）是一个关键组件，它的主要作用是根据模型在训练数据上计算得到的梯度，来更新模型的参数，使得模型的损失函数值逐渐减小，从而提高模型的性能。也就是说优化器用于结合损失函数、梯度来优化模型，常用的优化方法有SGD、Adam等。
- learn_optimizer.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 10:47
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
import time

try:
    # 加载 CIFAR - 10 训练数据集，并将图像转换为 Tensor 类型
    train_dataset = torchvision.datasets.CIFAR10("../testdataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    # 加载 CIFAR - 10 测试数据集
    test_dataset = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
except Exception as e:
    print(f"数据集加载失败: {e}")
else:
    class Yuzhao(nn.Module):
        def __init__(self):
            super(Yuzhao, self).__init__()
            self.model1 = Sequential(
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

    yuzhao = Yuzhao()
    loss_cross = nn.CrossEntropyLoss()
    # 创建优化器，使用更具描述性的名称
    optimizer = torch.optim.SGD(yuzhao.parameters(), lr=0.01)

    epochs = 20
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        yuzhao.train()  # 设置模型为训练模式
        for data in train_dataloader:
            imgs, targets = data
            outputs = yuzhao(imgs)
            # 计算损失
            results = loss_cross(outputs, targets)

            # 清空梯度
            optimizer.zero_grad()
            #反向传播
            results.backward()
            optimizer.step()

            running_loss += results.item()

        end_time = time.time()
        # 计算每个 epoch 的训练时间
        train_time = end_time - start_time

        # 打印每个 epoch 的损失和训练时间
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader):.4f}, Time: {train_time:.2f}s')

        # 在每个 epoch 结束后进行模型评估
        yuzhao.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                outputs = yuzhao(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # 打印每个 epoch 的测试准确率
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    print("训练结束")
```
> ```python
> # 计算损失
 >   1.results = loss_cross(outputs, targets)
>    # 清空梯度
 >   2.optimizer.zero_grad()
 >   #反向传播
>   3. results.backward()
 >   4.optimizer.step()
 >  ```
 >  ==这几行代码的顺序是必须严格按照给定顺序执行的==，下面详细解释每一步的含义以及为什么要按此顺序执行。
 > 标号1代码：这行代码使用预先定义好的交叉熵损失函数 loss_cross 来计算模型的预测输出 outputs 与真实标签 targets 之间的损失值 results。损失函数衡量了模型预测结果与真实情况之间的差异程度，在训练过程中，我们的目标就是通过不断调整模型的参数，使得这个损失值尽可能小。
 >  标号2代码：在 PyTorch 中，每次调用 backward() 方法进行反向传播时，计算得到的梯度会累加到模型参数的 .grad 属性中。如果不手动清空梯度，那么在每次迭代时，梯度会不断累积，导致梯度值异常，模型无法正常训练。因此，在每次反向传播之前，需要调用 optimizer.zero_grad() 方法将模型参数的梯度清零。
 >   标号3代码：行代码执行反向传播算法，根据损失值 results 计算模型中每个可训练参数的梯度。反向传播是基于链式法则，从损失函数开始，逐步计算每个参数对损失的偏导数，这些偏导数就是梯度，它们表示了参数在当前状态下对损失的影响程度。
 >    标号4代码：调用优化器的 step() 方法，根据之前计算得到的梯度来更新模型的参数。优化器（如随机梯度下降 SGD、Adam 等）会根据预设的优化策略（如学习率、动量等）对参数进行更新，使得损失值朝着减小的方向变化。
 >    </br>
 >    </br>
 >    ==总结==
 >    这几个步骤的顺序是固定的，原因如下：
先计算损失：只有先计算出损失值，才能明确模型当前的预测结果与真实标签之间的差距，为后续的反向传播和参数更新提供依据。
再清空梯度：如果在反向传播之后再清空梯度，那么本次计算得到的梯度就会被清空，无法用于参数更新；如果不清空梯度，梯度会不断累积，导致模型无法收敛。
接着反向传播：通过反向传播计算出梯度后，才能知道每个参数需要朝着哪个方向进行调整，以减小损失。
最后更新参数：在得到梯度之后，使用优化器根据梯度对参数进行更新，完成一次模型训练的迭代。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dc1442c798d54412b76251912162f3b5.png)


# 16. VGG16为例对现有的网络进行使用与修改

- VGG16_use_modify.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:04
# @Author : Yuzhao Li
import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

# 由于 ImageNet 数据集版权问题，通常需要手动下载和配置，这里注释掉
# train_data = torchvision.datasets.ImageNet("../data_ImageNet", split='train', transform=torchvision.transforms.ToTensor(), download=True)

# 创建不加载预训练权重的 VGG16 模型
vgg16_false = torchvision.models.vgg16(weights=None)
# 创建加载预训练权重的 VGG16 模型
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

# 查看不加载预训练权重的 VGG16 模型架构
print("不加载预训练权重的 VGG16 模型架构：")
print(vgg16_false)
print("\n")

#在整个网络后加模型
# 迁移学习，在 vgg16 后面加一个线性层（这里注释掉示例代码，仅展示思路）
# vgg16_false.add_module('add_linear', nn.Linear(1000, 10))
# print("在整个模型后面添加线性层后的 VGG16 模型架构：")
# print(vgg16_false)

#再VGG16写好的模块中加模型
# 在 vgg16 的 classifier 模块后面加一个线性层（这里注释掉示例代码，仅展示思路）
# vgg16_false.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print("在 classifier 模块后面添加线性层后的 VGG16 模型架构：")
# print(vgg16_false)

#修改指定的网络层
# 修改 vgg16 模型的 classifier 模块中的第 6 个线性层
vgg16_false.classifier[6] = nn.Linear(4096, 100)#本来是1000类
print("修改 classifier 模块中最后一个线性层后的 VGG16 模型架构：")
print(vgg16_false)

print("ok!")
```

会自动下载权重
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1156b774dce044b2b9662bdd3b8d9416.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/63cda0d5053049139c2861d80f5d9056.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7705ba1ac230445b8ed42c0e29de3504.png)
>torchvision.models 是 PyTorch 中 torchvision 库的一个重要模块，它提供了一系列预训练的深度学习模型，涵盖了图像分类、目标检测、语义分割、实例分割等多个计算机视觉任务。这些预训练模型在大规模数据集（如 ImageNet）上进行了训练，具有良好的特征提取能力，可以帮助开发者快速搭建和训练自己的模型，尤其在迁移学习场景中非常有用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b047b0d3f6cb4a80ba297e771799ebe5.png)




# 17. 模型的保存和加载
>在深度学习中，模型的保存和加载是非常重要的操作。保存模型可以方便我们在训练过程中保存中间结果、后续继续训练或者部署模型；加载模型则允许我们使用之前训练好的模型进行预测或者微调。
>==保存和加载整个模型（包括模型结构和参数）==
>保存：使用 torch.save() 函数将整个模型对象保存到文件中。
加载：使用 torch.load() 函数从文件中加载整个模型对象。
==保存和加载模型的参数（state_dict）==
保存：使用 torch.save() 函数将模型的 state_dict（一个包含模型所有可学习参数的字典）保存到文件中。
加载：先创建一个与原模型结构相同的模型实例，然后使用 load_state_dict() 方法将保存的参数加载到模型中。
## 17.1 保存
- learn_savemodel.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:19
# @Author : Yuzhao Li
import torch
import torchvision

# 导入 PyTorch 的神经网络模块
from torch import nn
# 导入二维卷积层类，用于构建自定义神经网络
from torch.nn import Conv2d

# 加载预训练的 VGG16 模型，但不加载预训练权重
vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式 1：保存模型及参数
# torch.save() 函数用于将对象保存到指定的文件中
# 这里将整个 VGG16 模型（包括模型结构和参数）保存到名为 'vgg16_method1.pth' 的文件中
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式 2：只保存参数
# vgg16.state_dict() 返回一个包含模型所有参数的字典
# 将这个字典保存到名为 'vgg16_method2.pth' 的文件中
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 注意事项说明：
# 使用方法 1 保存自己写的网络时，在不同环境加载可能会有问题。
# 需要确保加载环境中能访问到该网络类的定义，比如把网络类代码复制过去，或者使用 'from model_save import *' 导入
class Yuzhao(nn.Module):
    # 自定义神经网络类 Yuzhao，继承自 nn.Module
    def __init__(self, *args, **kwargs) -> None:
        # 调用父类 nn.Module 的构造函数
        super().__init__(*args, **kwargs)
        # 定义一个二维卷积层，输入通道数为 3，输出通道数为 32，卷积核大小为 5x5
        self.conv1 = Conv2d(3, 32, 5)

    def forward(self, x):
        # 定义前向传播过程
        # 将输入数据 x 通过卷积层 self.conv1 进行处理
        x = self.conv1(x)
        return x

# 创建自定义神经网络 Yuzhao 的实例
yuzhao = Yuzhao()
# 使用保存方式 1 将自定义模型（包括结构和参数）保存到名为 'yuzhao_method1.pth' 的文件中
torch.save(yuzhao, "yuzhao_method1.pth")
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6b533b087f3847489850c6496e63d958.png)

## 17.2 加载
>这段代码主要演示了在 PyTorch 中如何加载保存好的模型，包括加载预训练的 VGG16 模型和自定义的 Yuzhao 模型。加载模型有两种常见方式：加载整个模型和只加载模型参数。对于自定义模型，在加载时需要确保模型类的定义在当前环境中可用，否则会出现加载错误，代码中给出了两种解决该问题的方法。
- learn_loadmodel.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:22
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

# # 模型加载方法 1 - 对应保存方法 1
# # 此方法直接从文件中加载整个模型（包括模型结构和参数）
# # 加载保存的 VGG16 整个模型
# model1 = torch.load('./vgg16_method1.pth')
# # 打印加载后的模型结构
# print(model1)

# # # 加载方法 2
# # 这种方式只加载模型的参数
# # 直接加载保存的 VGG16 参数，此时得到的只是参数字典
# model2 = torch.load('./vgg16_method2.pth')
# # 打印参数字典
# print(model2)

# # 模型加载方法 2 - 对应保存方法 2
# # 若要完整加载网络，需先重新创建网络结构
# # 重新创建一个未加载预训练权重的 VGG16 模型
# vgg16 = torchvision.models.vgg16(weights=None)
# # 使用 load_state_dict 方法将保存的参数加载到新创建的 VGG16 模型中
# vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
# # 打印完整加载参数后的 VGG16 模型结构
# print(vgg16)


# 陷阱说明：
# # 若直接使用下面这行代码加载自定义模型，可能会出错
# # 因为在不同环境中加载自定义模型时，需要确保模型类的定义存在
# yuzhao1 = torch.load("yuzhao_method1.pth")
# print(yuzhao1)

# # 方法 1：手动定义模型类来解决加载自定义模型的问题
# # 定义自定义模型类 Yuzhao，继承自 nn.Module
# class Yuzhao(nn.Module):
#     # 类的初始化方法
#     def __init__(self, *args, **kwargs) -> None:
#         # 调用父类的初始化方法
#         super().__init__(*args, **kwargs)
#         # 定义一个二维卷积层，输入通道为 3，输出通道为 32，卷积核大小为 5
#         self.conv1 = Conv2d(3, 32, 5)
#
#     # 定义前向传播方法
#     def forward(self, x):
#         # 将输入数据 x 通过卷积层进行处理
#         x = self.conv1(x)
#         return x
#
# # 现在可以正常加载自定义模型了
# # 因为已经在当前环境中定义了 Yuzhao 类
# yuzhao2 = torch.load("yuzhao_method1.pth")
# # 打印加载后的自定义模型结构
# print(yuzhao2)


# 方法 2：直接把保存模型的文件引入
# 例如使用 from learn_savemodel import *
# 若在 learn_model_save 文件中定义了 Yuzhao 类，通过这种导入方式
# 可以在当前环境中使用该类，从而正确加载保存的自定义模型
from learn_savemodel import *
yuzhao3= torch.load("yuzhao_method1.pth")
print(yuzhao3)
```
方法1加载的完整模型
- print(model1)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97e07be55f2541619ee1fc4c3b967ce3.png)
方法2只加载了模型参数
- print(model2)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b38463f37d0748dcae02ee217aee0af2.png)
方法2加载参数后又加载模型
- print(vgg16)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/60e6a5d5edd94c58b6b07cb11a2b3789.png)
加载保存的自己的网络
- print(yuzhao1)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6756e89439d948a39b578a2cc2bc5fd7.png)
加载保存的自己的网络->重新定义一遍网络
- print(yuzhao2)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4db04ac620294f57becf5c9ec4d0af04.png)

加载保存的自己的网络->导入定义的网络
- print(yuzhao3)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bda3cefd31a240e9a2371c336b0e7923.png)

# 18. 完整的模型训练流程
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61d00008125e458eb8b7d9b0a2885923.png)

>使用Cifar10结构作为实例
## 18.1 只有训练 （Only train）
- learn_Cifar10_train.py
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/18 11:43
# @Author : Yuzhao Li
import torch
import torchvision
from torch import nn
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.data import DataLoader
# 从 model.py 文件中导入所有定义的类和函数
# 这里假设 learn_savemodel.py 文件中定义了名为 Yuzhao 的神经网络类

class Yuzhao(nn.Module):
    def __init__(self):
        super(Yuzhao, self).__init__()
        self.model1 = Sequential(
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

# 准备数据集
# 加载 CIFAR - 10 训练数据集
# "../testdataset" 是数据集存储的路径
# train=True 表示加载训练集
# transform=torchvision.transforms.ToTensor() 将图像数据转换为 Tensor 类型
# download=True 若数据集不存在则进行下载
train_data = torchvision.datasets.CIFAR10("../testdataset", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
# 加载 CIFAR - 10 测试数据集
test_data = torchvision.datasets.CIFAR10("../testdataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

# len() 函数用于获取数据集的样本数量
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
# 从 learn_savemodel.py 文件中引入的 Yuzhao 类创建一个神经网络实例
yuzhao = Yuzhao()

# 定义损失函数
# 使用交叉熵损失函数，常用于分类任务
loss_fn = nn.CrossEntropyLoss()

# 定义优化器的超参数
# 学习率设置为 0.01
learning_rate = 1e-2
# 使用随机梯度下降（SGD）优化器
# yuzhao.parameters() 获取神经网络的可训练参数
# lr=learning_rate 设置学习率
optimizer = torch.optim.SGD(yuzhao.parameters(), lr=learning_rate)

# 设置网络训练的一些参数
# 记录训练的总步数
total_train_step = 0
# 记录测试的总步数
total_test_step = 0
# 定义训练的总轮数
epoch = 10

# 开始训练过程
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))

    # 遍历训练数据加载器中的每个批次
    for data in train_dataloader:
        # 解包每个批次的数据，得到图像数据和对应的标签
        imgs, targets = data
        # 将图像数据输入到神经网络中进行前向传播，得到预测输出
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
        # 打印当前训练步数和对应的损失值
        print("训练次数：{}，Loss:{}".format(total_train_step, loss))
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/882b47e447274825b88639d203c44a97.png)


## 18.2 训练+测试（train+test）

- learn_Cifar10_tarin_test.py
```python
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
    # 注释掉的代码，torch.train() 实际应为 model.train()，用于将模型设置为训练模式
    # 此模式对部分特殊层（如 Dropout 等）有影响
    # torch.train()

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
    # torch.eval() 实际应为 model.eval()，用于将模型设置为评估模式
    # 此模式对部分特殊层（如 Dropout 等）有影响
    torch.eval()

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
        writer.add_scalar("test_auccracy", total_accuracy, total_test_step)
        # 测试步数加 1
        total_test_step += 1

    # 保存模型
    # 注释掉的代码，将整个模型保存到文件中，文件名包含当前轮数
    torch.save(yuzhao,"yuzhao_{}.pth".format(i+1))

# 关闭 SummaryWriter 对象，确保所有数据都写入日志文件
writer.close()

```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c60c16d89b0540fcaae8b6337d0fc6f3.png)


>实现了一个基于 CIFAR - 10 数据集的卷积神经网络的训练和测试过程。==主要步骤包括：==
数据集准备：使用 torchvision 加载 CIFAR - 10 训练集和测试集，并将图像数据转换为 Tensor 类型。
数据加载：使用 DataLoader 对数据集进行批量加载。
模型搭建：定义了一个名为 Yuzhao 的卷积神经网络模型。
损失函数和优化器定义：使用交叉熵损失函数和随机梯度下降（SGD）优化器。
训练过程：进行多个轮次的训练，每个轮次中遍历训练数据加载器，进行前向传播、损失计算、反向传播和参数更新，并记录训练损失和训练时间。
测试过程：每个训练轮次结束后，使用测试数据集对模型进行评估，计算测试集上的损失和准确率，并记录到 TensorBoard 日志中。
日志记录：使用 SummaryWriter 将训练和测试过程中的损失和准确率信息写入 TensorBoard 日志。

# 19. 使用GPU训练
可以使用cuda加速的部分
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c90d79eb6fad458b99ee497cf9f99672.png)
## 19.1 自己电脑GPU cuda()方法
>整体流程与上面一样，上面已经默认加上cuda加速
```python
#模型
if torch.cuda.is_available():
    yuzhao=yuzhao.cuda()

#损失函数
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

#数据
for data in train_dataloader:
    start_time = time.time()  # 记录每个批次开始的时间
    imgs, targets = data
    if torch.cuda.is_available():
	    imgs = imgs.cuda()
	    targets = targets.cuda()
```
 ## 19.2 自己电脑gpu to方法
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/caa81e0a94cb4129a6604e45eb534e38.png)
>使用torch.device("硬件")创建硬件对象
>torch.device("gpu") #gpu实例
>torch.device("cuda") #第一张显卡，默认为第一张
>torch.device("cuda:0") # 第一张显卡，序号从0开始，0表示第一张
>torch.device("cuda:1")#  第二张显卡，序号从0开始，1表示第二张
>(数据/模型/损失函数).to(创建的硬件实例) #使用gpu加速训练对应部分

```python
#定义训练的设备
device=torch.device("cuda")

#模型
yuzhao=Yuzhao()
yuzhao.to(device)

#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn.to(device)

for data in train_dataloader:
    imgs,targets=data
    imgs=imgs.to(device)
    targets=targets.to(device)

```

# 20. 完整的模型验证- 利用已经训练好的模型提供输入
>这段代码的主要功能是加载一张图片，对其进行预处理，然后尝试加载一个预训练好的 PyTorch 模型。具体步骤如下：
1.定义神经网络模型：定义了一个名为 Yuzhao 的卷积神经网络模型，包含卷积层、池化层、展平层和全连接层。
2.加载图片并预处理：使用 PIL 库打开一张图片，将其转换为 RGB 格式，然后使用 torchvision.transforms 对图片进行调整大小和转换为张量的操作。
3.加载预训练模型：尝试从文件 yuzhao_3.pth 中加载预训练好的模型，并将其映射到 CPU 上。
>4. 模型推理
在成功加载图片和预训练模型之后，就可以进行模型推理了。推理过程就是将预处理好的图片输入到加载的模型中，让模型对图片进行分析并给出预测结果。
4.1设置模型为评估模式：在进行推理之前，需要将模型设置为评估模式，通过调用 model.eval() 方法来实现。这是因为有些层（如 Dropout、BatchNorm 等）在训练和推理时的行为是不同的，设置为评估模式可以确保这些层在推理时使用正确的参数和逻辑。
4.2关闭梯度计算：为了节省计算资源和内存，在推理阶段不需要计算梯度。可以使用 torch.no_grad() 上下文管理器来临时关闭梯度计算。在这个上下文环境中进行的所有操作都不会记录梯度信息。
4.3输入图片进行推理：将预处理好的图片输入到模型中，调用 model(image) 进行前向传播，得到模型的输出。这个输出通常是一个张量，其每个元素代表图片属于不同类别的得分或者概率。
>5. 处理预测结果
得到模型的输出后，需要对其进行处理以得到最终的预测类别。
找出最大得分对应的索引：使用 torch.max() 函数在模型输出的张量中找出得分最高的元素对应的索引。这个索引就代表了模型预测的图片所属的类别。
映射索引到类别名称：在分类任务中，通常会有一个类别名称列表，每个索引对应一个具体的类别名称。通过将得到的索引作为下标，从类别名称列表中取出对应的类别名称，就可以得到最终的预测结果。
>6. 输出预测结果
最后，将处理好的预测结果输出，让用户知道模型对输入图片的分类结果。可以使用 print() 函数将预测结果打印到控制台，或者将其保存到文件中，以便后续分析和使用。
- learn_valmodel.py
```python
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
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bff496506d814eb7996b54808b58683f.png)

到此，PyTorch基础入门学习就算结束了，很高兴你我能学到这里，这些只是基础使用，如有更深的需求请持续学习。好了，技能已经学会，去打怪吧！！



