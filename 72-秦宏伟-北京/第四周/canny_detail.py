#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Canny Detail边缘检测"""
import numpy as np
from sklearn.datasets._base import load_iris
import matplotlib.pyplot as plt
import cv2
import math


"""
2、高斯滤波
dim 高斯滤波滤波器大小，默认为奇数，如果为偶数，则+1变为奇数
siggma 高斯滤波的波形宽度，siggma越大，波形越宽，平滑效果越明显；越小，波形越高，滤波效果不明显
"""
def gx_filter(gray,dim,siggma):
    dim = (dim // 2) * 2 + 1
    center = dim // 2
    #siggma取绝对值
    siggma =abs(siggma)
    # 计算二维高斯函数
    filer_gx = np.zeros([dim,dim])
    n1 = 1/(2*math.pi*siggma**2)
    n2 = -1/(2*siggma**2)
    for i in range(dim):
        for j in range(dim):
            filer_gx[i,j] = n1* math.exp(n2*((center-i)**2+(center-j)**2))
    #sum各个权值
    sum = filer_gx.sum()
    filer_gx = filer_gx / sum
    #灰度图填充，步长为1，pad = (f-1)/2,f为卷积核大小，即dim
    tmp = dim//2
    gray_pad = np.pad(gray, ((tmp, tmp), (tmp, tmp)), 'constant')
    #对灰度图进行高斯过滤
    fileter_gray = np.zeros([gray.shape[0],gray.shape[1]])
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            fileter_gray[i,j] = np.sum(gray_pad[i:i+dim,j:j+dim]*filer_gx)
    return fileter_gray

"""
3、对图片进行sobel算子过滤
"""
def sobel(gray):
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    image_x = np.zeros(gray.shape)
    image_y = np.zeros(gray.shape)
    image_tidu = np.zeros(gray.shape)
    image_pad = np.pad(gray,((1,1),(1,1)),'constant')
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            image_x[i,j] = np.sum(image_pad[i:i+3,j:j+3]*sobel_x)
            image_y[i,j] = np.sum(image_pad[i:i+3,j:j+3]*sobel_y)
            image_tidu[i,j] = np.sqrt(image_x[i,j]**2+image_y[i,j]**2)
    image_x[image_x==0] = 0.0000000001
    image_y[image_y == 0] = 0.0000000001
    angel = image_y/image_x
    return image_tidu,angel

"""
对梯度幅值进行非极大值抑制
1、将当前像素的梯度强度与沿正负梯度方向上的两个像素进行比较。
2、如果当前像素的梯度强度与另外两个像素相比最大，则该像素点保留为边缘点，否则
该像素点将被抑制（灰度值置为0）。
"""
def non_max_filter(image_tidu,angel):
    image_filter = np.zeros(image_tidu.shape)
    for i in range(1,image_tidu.shape[0]-1):
        for j in range(1,image_tidu.shape[1]-1):
            # dTemp1 = 0
            # dTemp2 = 0
            if(angel[i,j])>1:
                #梯度方向更靠近Y轴方向,并且gy与gx的方向相同
                weight = 1/abs(angel[i,j])
                g1 = image_tidu[i-1,j-1]
                g2 = image_tidu[i,j-1]
                g3 = image_tidu[i+1,j+1]
                g4 = image_tidu[i,j+1]
            elif(angel[i,j])<-1:
                # 梯度方向更靠近Y轴方向,并且gy与gx的方向相反
                weight = 1/abs(angel[i, j])
                g1 = image_tidu[i + 1, j - 1]
                g2 = image_tidu[i, j - 1]
                g3 = image_tidu[i - 1, j + 1]
                g4 = image_tidu[i, j + 1]
            elif 0 < (angel[i,j]):
                # 梯度方向更靠近X轴方向,并且gy与gx的方向相同
                weight = abs(angel[i, j])
                g1 = image_tidu[i - 1, j + 1]
                g2 = image_tidu[i - 1, j]
                g3 = image_tidu[i + 1, j - 1]
                g4 = image_tidu[i + 1, j]
            elif (angel[i,j]) <0:
                # 梯度方向更靠近X轴方向,并且gy与gx的方向相反
                weight = abs(angel[i, j])
                g1 = image_tidu[i - 1, j - 1]
                g2 = image_tidu[i - 1, j]
                g3 = image_tidu[i + 1, j + 1]
                g4 = image_tidu[i + 1, j]
            dTemp1 = weight * g1 + (1 - weight) * g2;
            dTemp2 = weight * g3 + (1 - weight) * g4;
            if image_tidu[i,j]> dTemp1 and image_tidu[i,j]> dTemp2:
                image_filter[i,j] = image_tidu[i,j]
    return image_filter

def double_threadhold_filter(min_threadhold,max_threadhold,image):
    image_filter = np.zeros(image.shape)
    zhan = []
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i,j]>=max_threadhold:
                image_filter[i,j] = 255
                zhan.append([i, j])
            elif image[i,j]>=min_threadhold:
                image_filter[i,j] = image[i,j]

    while len(zhan) >0:
        [i,j] = zhan.pop()
        for tmp_i in [i-1,i+1]:
            for tmp_j in [j-1,j+1]:
                if tmp_i ==i and tmp_j ==j:
                    continue
                if image_filter[tmp_i,tmp_j]>=min_threadhold and image_filter[tmp_i,tmp_j]<=max_threadhold:
                    image_filter[tmp_i,tmp_j] = 255
                    zhan.append([tmp_i,tmp_j])
    return image_filter
"""
基于灰度图，进行Canny边缘检测
"""
def main(fileName):
    # 1、转为灰度图
    img = cv2.imread(fileName, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #高斯滤波
    dim = 4
    siggma = 1.5
    filter_gray = gx_filter(gray, dim, siggma)
    # 展示灰度图
    # plt.imshow(gray,cmap='gray')
    # plt.axis('off')
    # plt.show()
    # 展示高斯滤波后的图
    plt.figure(1)
    plt.imshow(filter_gray,cmap='gray')
    plt.axis('off')
    # plt.show()
    image_tidu,angel = sobel(filter_gray)
    plt.figure(2)
    plt.imshow(image_tidu, cmap='gray')
    plt.axis('off')
    # plt.show()
    img_yizhi = non_max_filter(image_tidu,angel)
    plt.figure(3)
    plt.imshow(img_yizhi, cmap='gray')
    plt.axis('off')
    # plt.show()
    min_threadhold = np.mean(img_yizhi)
    max_threadhold = 3 * min_threadhold
    img_canny = double_threadhold_filter(min_threadhold,max_threadhold,img_yizhi)
    plt.figure(4)
    plt.imshow(img_canny, cmap='gray')
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(filter_gray,cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(image_tidu, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(img_yizhi, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(img_canny, cmap='gray')
    plt.show()
if __name__ == '__main__':
    main("lenna.png")