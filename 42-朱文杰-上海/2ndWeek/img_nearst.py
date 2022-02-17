# -*- coding: utf-8 -*-
"""
Proximity interpolation

Date:2021.11.27
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.color import rgb2gray

test_img = "lenna.png"
dstHeight = 800
dstWidth = 800

def NearstImg(imgSource):
    """

    :param imgSource:
    :return:
    """

    imgSourceH, imgSourceW, channels = imgSource.shape
    imgNearst = np.zeros((dstHeight, dstWidth, channels), imgSource.dtype)
    imgH = dstHeight / imgSourceH
    imgW = dstWidth / imgSourceW
    for i in range(dstHeight):
        for j in range(dstWidth):
            x = int(i / imgH)
            y = int(j / imgW)
            imgNearst[i, j] = imgSource[x, y]
    return imgNearst

if __name__ == '__main__':
    img_source = cv2.imread(test_img)
    img_nearst = NearstImg(img_source)
    cv2.imshow('img_source', img_source)
    cv2.imshow('img_nearst', img_nearst)
    cv2.waitKey(0)