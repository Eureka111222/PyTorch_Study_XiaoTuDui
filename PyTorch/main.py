#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/02/17 21:04
# @Author : Yuzhao Li
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  # 输出为True，则安装成功
dir(torch)