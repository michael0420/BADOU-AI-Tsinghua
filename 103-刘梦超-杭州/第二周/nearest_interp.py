#!/usr/bin/env python 
# coding:utf-8
import cv2
import numpy as np


# 处理放大后的图片
def del_img_function(img):
    # 原图的行,列,通道数
    height, width, channels = img.shape
    # 创建一个800*800的图片
    dst_h = 800
    dst_w = 800
    empty_img = np.zeros((dst_h, dst_h, channels), img.dtype)
    # 原图和目标图的比例
    ratio_h = dst_h / height
    ratio_w = dst_w / width
    # 遍历矩阵
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i / ratio_h)
            y = int(j / ratio_w)
            empty_img[i, j] = img[x, y]
    return empty_img


# 读入原图
img = cv2.imread("lenna.png")
cv2.imshow("src_img", img)
print(img.shape)

# 获得放大后的图片
del_img = del_img_function(img)
cv2.imshow("dst_img", del_img)
print(del_img.shape)

# 显示最终结果
cv2.waitKey(0)
cv2.destroyWindow()