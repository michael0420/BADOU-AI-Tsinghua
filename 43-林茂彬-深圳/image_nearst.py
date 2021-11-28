# -*- coding: utf-8 -*-
"""
2021-11-25 临近插值法
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2



# 临近插值法
def img_nearst(img):
    img_source = cv2.imread("lenna.png") # 导入图片
#print(img_gray.shape) #得到图片的（行，列，通道）
    img_source_h,img_source_w,channels=img_source.shape
    img_nearst = np.zeros((800, 800,channels), img_source.dtype)
    img_h = 800 / img_source_h
    img_x = 800 / img_source_w
    for i in range(800):
        for j in range(800):
            x = int(i / img_h)
            y = int(j / img_x)
            img_nearst[i, j] = img[x, y]
    return img_nearst

if __name__ == '__main__':
    img_source = cv2.imread('lenna.png')
    img_nearst = img_nearst(img_source)
    cv2.imshow('img_source', img_source)
    cv2.imshow('img_nearst', img_nearst)
    cv2.waitKey(0)