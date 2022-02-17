#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 灰度图像直方图均衡化
def image_histogram_equalization(img):
    img_source=img
    img_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    img_equalizeHist = cv2.equalizeHist(img_gray)
    cv2.imshow("img_gray", img_gray)
    cv2.imshow("img_equalizeHist", img_equalizeHist)
    cv2.waitKey(0)

# 彩色图像直方图均衡化
def image_histogram_equalization_color(img):
    img_source = img
    (b, g, r) = cv2.split(img) # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    img_source_b = cv2.equalizeHist(b)
    img_source_g = cv2.equalizeHist(g)
    img_source_r = cv2.equalizeHist(r)
    # 合并每一个通道
    img_result = cv2.merge((img_source_b, img_source_g, img_source_r))
    cv2.imshow("img", img)
    cv2.imshow("img_result", img_result)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_source = cv2.imread('lenna.png')
    #image_histogram_equalization = image_histogram_equalization(img_source)
    image_histogram_equalization_color = image_histogram_equalization_color(img_source)
