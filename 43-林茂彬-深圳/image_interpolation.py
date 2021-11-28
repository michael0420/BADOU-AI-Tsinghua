# -*- coding: utf-8 -*-
"""
2021-11-25 双线性插值算法
# https://blog.csdn.net/qq_41375609/article/details/102650671 可参考这个blog
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2



# 双线性插值
def image_interpolation(img,img_dst): #导入原图及目标图的长宽
    img_source_h, img_source_w,channel = img.shape#获取原图片数值
    img_dst_h, img_dst_w = img_dst[1],img_dst[0]  # 获取目标图片数值
    print("img_source_h, img_source_w = ", img_source_h, img_source_w)
    print("img_dst_h, img_dst_w = ", img_dst_h, img_dst_w)
    if img_source_h == img_dst_h and img_source_w == img_dst_w:
        return img.copy()
    rate_y=img_source_h/img_dst_h #计算原表与目标表比例
    rate_x=img_source_w/img_dst_w
    img_dst = np.zeros((img_dst_h, img_dst_w,3), img_source.dtype) #创建空的目标表数组
    for i in range(3):
        for dst_y in range(img_dst_h):
            for dst_x in range(img_dst_w):
            #要通过双线性插值的方法计算出dst中每一个像素点的像素值，需要通过dst像素点的坐标对应到src图像当中的坐标，然后通过双线性插值的方法算出src中相应坐标的像素值。
                src_x = (dst_x + 0.5) * rate_x - 0.5
                src_y = (dst_y + 0.5) * rate_y - 0.5
            #找出原图平面内四个像素点的坐标
            #　对于一个目的像素，设置坐标通过反向变换得到的浮点坐标为(i + u, j + v)(其中i、j均为浮点坐标的整数部分，u、v为浮点坐标的小数部分，是取值[0, 1)区间的浮点数)，则这个像素得值
            #  f(i + u, j + v)
            #  可由原图像中坐标为(i, j)、(i + 1, j)、(i, j + 1)、(i + 1, j + 1)
            # 所对应的周围四个像素的值决定，即：f(i + u, j + v) = (1 - u)(1 - v)
            # f(i, j) + (1 - u)
            # vf(i, j + 1) + u(1 - v)
            # f(i + 1, j) + uvf(i + 1, j + 1)。其中f(i, j)
            #表示源图像(i, j)处的的像素值，以此类推。
            #注：比如坐标（1.3，1.4）= > （1 + 0.3，1 + 0.4），那么可由原图像中的坐标为（1，1）、（2，1），（1，2），（2，2）所对应的周围四个像素的值决定
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, img_source_w - 1) # src_w - 1 为了防止超出边界 比方到512时 src_x0 + 1 已经超了 （513）
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, img_source_h - 1)
            # 算出四个坐标后根据 单线性插值公式 分别求出两个 R点 y=（x1-x）*y0+(x-x0)*y1
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
            # 算出两个R点  在根据单线性插值公式 求出两个p点 p=（y1-y）*temp0+(y-y0)*temp1
                img_dst[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return img_dst
if __name__ == '__main__':
    img_source = cv2.imread('lenna.png')
    image_interpolation=image_interpolation(img_source,(800,800))
    cv2.imshow('img_source', img_source)
    cv2.imshow('image_interpolation', image_interpolation)
    cv2.waitKey(0)