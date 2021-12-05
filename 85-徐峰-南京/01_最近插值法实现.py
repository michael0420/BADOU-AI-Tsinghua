#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/11/22 23:35 
# ide： PyCharm


import cv2
import numpy as np


def nearestInterplot(img):
    #height, width, channels
    height, width, channels = img.shape

    #new image with 800 * 800
    #dtype 对图像会有影响
    empty_img = np.zeros((800, 800, channels), dtype=np.uint8)

    sh = height / 800
    sw = width / 800

    #formula
    """
    srcX = dstX * (srcWidth/dstWidth)
    srcY = dstY * (srcHeight/dstHeight)
    """

    #fill the empty_img
    for i in range(800):
        for j in range(800):
            x = int(i * sh)
            y = int(j * sw)
            empty_img[i, j] = img[x, y]
    return empty_img

img = cv2.imread("../../../BaiduNetdiskDownload/lenna.png")
zoom = nearestInterplot(img)
print(zoom.shape)
cv2.imshow("neraest interplot", zoom)
cv2.waitKey(0)
