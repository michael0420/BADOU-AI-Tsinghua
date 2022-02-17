#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/5 23:39 
# ide： PyCharm

import numpy as np
import matplotlib.pyplot as plt


#二维降一维
x = np.linspace(0 ,10, 200) + np.random.rand(200) / 5
y = 0.233 * x + np.random.rand(200) / 3

#归一化
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

#计算协方差矩阵
con_mat = np.cov(x, y)
#计算协方差矩阵的特征值
eigenvalues, egienvectors = np.linalg.eig(con_mat)
print(egienvectors)
print(eigenvalues)
k = 1 #降维后的目标维度
#对特征值进行排序，取前k个
topk = np.argsort(eigenvalues)[0:k]
print(topk)
#取前k个最大的特征值对应的特征向量与原矩阵进行相乘
data = np.stack((x, y), axis=-1)
print(data)
res = np.matmul(data, egienvectors[topk].T)
plt.plot(res[:, 0], res[:, 0].shape[0] * [1], '.')
plt.show()


