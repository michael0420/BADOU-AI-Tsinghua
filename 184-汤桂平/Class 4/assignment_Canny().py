# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/3/13

import numpy as np
import cv2
import math
from collections import deque

def Canny(img, sigma, high_threshold, low_threshold):
    # 1、图像灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2、高斯滤波
    # 求高斯核的尺寸
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim = dim + 1
    # 创建高斯核
    Gauss_filter = np.zeros([dim, dim])
    tmp = [i - dim//2 for i in range(dim)]
    n1 = 1/(2 * math.pi * sigma**2)
    n2 = -1/(2 * sigma**2)
    # 遍历求高斯核，利用高斯函数离散取样得到高斯核
    for i in range(dim):
        for j in range(dim):
            Gauss_filter[i, j] = n1 * math.exp(n2 * (tmp[i]**2 + tmp[j]**2))
    Gauss_filter = Gauss_filter / Gauss_filter.sum()
    dx, dy = img_gray.shape
    # 存储高斯平滑后的图像
    img_new = np.zeros(img_gray.shape)
    tmp = dim//2
    # 边缘填充后的二维图像数组
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gauss_filter)

    cv2.imshow('高斯平滑后的图像：', img_new)

    # 3、利用sobel求梯度（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1],[-2, 0, -2],[-1, 0, 1]])  # 水平卷积核
    sobel_kernel_y = np.array([[1, -2, 1],[0, 0, 0],[-1, -2, -1]])  # 垂直卷积核
    tidu_x = np.zeros(img_new.shape)  # 创建水平梯度图像数组
    tidu_y = np.zeros(img_new.shape)  # 创建垂直梯度图像数组
    tidu = np.zeros(img_new.shape)
    sobel_pad = np.pad(img_new, ((1, 1),  (1, 1)), 'constant')  # 边缘填充，水平方向和垂直方向边缘均一行，常量填充0
    for i in range(dx):
        for j in range(dy):
            tidu_x[i, j] = np.sum(sobel_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            tidu_y[i, j] = np.sum(sobel_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            tidu[i, j] = np.sqrt(tidu_x[i, j]**2 + tidu_y[i, j]**2)

    angle = tidu_y / tidu_x    # 求梯度方向，为非极大值抑制做准备
    cv2.imshow('X方向的梯度图', tidu_x)
    cv2.imshow('y方向的梯度图', tidu_y)
    cv2.imshow('梯度图', tidu)

    # 非极大值抑制
    tidu_yizhi = np.zeros(tidu.shape)  # 创建抑制后的图像
    for i in range(1, dx - 1):  # 遍历除边缘外的所有像素
        for j in range(1, dy - 1):
            flag = True    # 创建标签，Ture则保留原始像素值， False则置为0
            tmp = tidu[i-1:i+2, j-1:j+2]  # 创建包含邻域8像素在内的3*3矩阵
            # 梯度方向有4种情况，利用线性插值法判断是否抑制
            if angle[i, j] < -1:
                n1 = tmp[0, 1] + (tmp[0, 1] - tmp[0, 0]) / angle[i, j]
                n2 = tmp[2, 1] + (tmp[2, 1] - tmp[2, 2]) / angle[i, j]
                if not (tmp[1, 1] > n1 and tmp[1, 1] > n2):
                    flag = False
            elif angle[i, j] > 1:
                n1 = tmp[0, 1] + (tmp[0, 1] - tmp[0, 2]) / angle[i, j]
                n2 = tmp[2, 1] + (tmp[2, 1] - tmp[2, 0]) / angle[i, j]
                if not (tmp[1, 1] > n1 and tmp[1, 1] > n2):
                    flag = False
            elif angle[i, j] < 0:
                n1 = tmp[1, 0] + (tmp[1, 0] - tmp[0, 0]) * angle[i, j]
                n2 = tmp[1, 2] + (tmp[1, 2] - tmp[2, 2]) * angle[i, j]
                if not (tmp[1, 1] > n1 and tmp[1, 1] > n2):
                    flag = False
            elif angle[i, j] > 0:
                n1 = tmp[1, 2] + (tmp[1, 2] - tmp[0, 2]) * angle[i, j]
                n2 = tmp[1, 0] + (tmp[1, 0] - tmp[2, 0]) * angle[i, j]
                if not (tmp[1, 1] > n1 and tmp[1, 1] > n2):
                    flag = False
            if flag:
                tidu_yizhi[i, j] = tidu[i, j]
    cv2.imshow('非极大值抑制后的图像：', tidu_yizhi)

    # 双阈值检测
    zhan = deque()  # 使用collections.deque实现栈
    for i in range(1, tidu_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, tidu_yizhi.shape[1] - 1):
            if tidu_yizhi[i, j] >= high_threshold:  # 取一定是边缘的点，并置为255，且把该像素点的坐标进栈
                tidu_yizhi[i, j] = 255
                zhan.append([i, j])
            elif tidu_yizhi[i, j] <= low_threshold:  # 低于低阈值的像素点置为0
                tidu_yizhi[i, j] = 0

    while not len(zhan) == 0:   # 对于一定是边缘的点（强边缘），以它为中心取其邻域8像素，然后进行遍历，只要是弱边缘，即认定其为边缘
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = tidu_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_threshold) and (a[0, 0] > low_threshold):
            tidu_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈，作为一个新的强边缘
        if (a[0, 1] < high_threshold) and (a[0, 1] > low_threshold):
            tidu_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_threshold) and (a[0, 2] > low_threshold):
            tidu_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_threshold) and (a[1, 0] > low_threshold):
            tidu_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_threshold) and (a[1, 2] > low_threshold):
            tidu_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_threshold) and (a[2, 0] > low_threshold):
            tidu_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_threshold) and (a[2, 1] > low_threshold):
            tidu_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_threshold) and (a[2, 2] > low_threshold):
            tidu_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(tidu_yizhi.shape[0]):
        for j in range(tidu_yizhi.shape[1]):
            if tidu_yizhi[i, j] != 0 and tidu_yizhi[i, j] != 255:
                tidu_yizhi[i, j] = 0
    cv2.imshow('Canny处理后的最终效果图：', tidu_yizhi)
    return tidu_yizhi

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_Canny = Canny(img, 0.5, 100, 50)
    cv2.imshow('Canny处理后的最终效果图：', img_Canny)
    cv2.waitKey(0)





