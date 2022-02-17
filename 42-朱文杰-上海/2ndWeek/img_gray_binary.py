# -*- coding: utf-8 -*-
"""
Grayscale image & binary image

Date:2021.11.27
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.color import rgb2gray

test_img = "lenna.png"

def GrayscaleImg(imgSource):
    """
    :param imgSource:
    :return:
    """
    imgSourceH,imgSourceW = imgSource.shape[0:2]
    imgGray = np.zeros([imgSourceH, imgSourceW], imgSource.dtype)

    for i in range(imgSourceH):
        for j in range(imgSourceW):
            a=imgSource[i, j]
            imgGray[i, j] = int(a[0] * 0.11 + a[1] * 0.59 + a[2] * 0.3)
    return imgGray

def BinaryImg(imgSource):
    """
    :param imgSource:
    :return:

    """
    imgGray = rgb2gray(imgSource)
    imgBinary = np.where(imgGray < 0.5, 0, 255).astype(np.uint8)
    return imgBinary


if __name__ == '__main__':
    img_source = cv2.imread(test_img)
    img_gray = GrayscaleImg(img_source)
    img_binary = BinaryImg(img_source)
    cv2.imshow('img_source', img_source)
    cv2.imshow('img_gray', img_gray)
    cv2.imshow('img_binary', img_binary)
    cv2.waitKey(0)