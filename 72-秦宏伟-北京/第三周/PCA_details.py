#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直方图均衡化"""
import numpy as np
from sklearn.datasets._base import load_iris
import matplotlib.pyplot as plt
"""
PCA降维详细版1，不进行数据中心化，直接求协方差矩阵
"""

"""
求向量的协方差
输入 二维数组，向量为N*2
"""
def cov(data_x,data_y):
    if len(data_x) != len(data_y):
        return None
    avg_x = np.average(data_x)
    avg_y = np.average(data_y)
    cov_xy = 0
    for i in range(len(data_x)):
        cov_xy += (data_x[i]-avg_x)*(data_y[i]-avg_y)
    cov_xy = cov_xy/(len(data_x)-1)
    return cov_xy

"""
求协方差矩阵
输入 M*N维数组
"""
def cov_matrix(data):
    N = data.shape[1]
    cov_matrix = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            if cov_matrix[j,i] !=0:
                cov_matrix[i,j] = cov_matrix[j,i]
                continue
            Cij = cov(data[:,i],data[:,j])
            cov_matrix[i,j] = Cij
    return cov_matrix

"""
求特征向量
"""
def det(cov_matrix):
    det_w,det_v = np.linalg.eig(cov_matrix)
    return det_w,det_v

"""
矩阵中心化
"""
def centerlize(data):
    avg = np.zeros(data.shape[1])
    for j in range(data.shape[1]):
        avg[j] = np.average(data[:,j])
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            data[i,j] = data[i,j]-avg[j]
    return data

"""
1、求协方差矩阵
2、求特征向量和特征值
3、求新的特征空间
输入data M*N，N为特征
"""
def main(data,label):
    #矩阵中心化
    data = centerlize(data)
    # 求协方差矩阵
    cov_array = cov_matrix(data)
    #det_w为特征向量，det_v为特征值
    det_w,det_v = det(cov_array)
    #对特征向量进行排序，取前K个特征值,argsort采用的是升序，因此用-1乘特征值，输出的结果为倒序的索引值
    sort_index = np.argsort(-1*det_w)
    k=2
    top_index = sort_index[:k]
    #获取前k个特征向量
    top_det = det_v[:,top_index]
    #对特征向量和输入矩阵做卷积
    result = np.dot(data,top_det)
    #展示效果
    show(result,label)

def show(reduced_x,y):
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()

if __name__ == '__main__':
    #加载数据
    data = np.array([[10, 15, 29, 49],
                  [15, 46, 13, 46],
                  [23, 21, 30, 46],
                  [11, 9,  35, 9],
                  [42, 45, 11, 45],
                  [9,  48, 5, 5],
                  [11, 21, 14, 14],
                  [8,  5,  15,  15],
                  [11, 12, 21, 21],
                  [21, 20, 25, 20]])

    x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签
    main(x, y)
