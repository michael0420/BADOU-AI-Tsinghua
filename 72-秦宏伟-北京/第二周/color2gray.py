#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""彩色图像转为灰度图和二值图"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def color2gray(B,G,R):
    # gray = np.empty([B.shape[0],B.shape[1]])
    gray = np.zeros([B.shape[0],B.shape[1]],np.uint8)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            gray[i][j] = 0.3*R[i][j]+ 0.59*G[i][j] + 0.11*B[i][j]
    return gray

def cloor2bit(B,G,R):
    bit = np.zeros([B.shape[0],B.shape[1]],np.uint8)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if(int(0.3*R[i][j]+ 0.59*G[i][j] + 0.11*B[i][j])>128):
                bit[i][j] = 255
            else:
                bit[i][j] = 0
    return bit


if __name__ == '__main__':
    image = cv.imread('lenna.png')
    (B,G,R) = cv.split(image)
    image_gray = color2gray(B,G,R)
    image_bit = cloor2bit(B,G,R)

    plt.imshow(image_gray,cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(image_bit, cmap='gray')
    plt.axis('off')
    plt.show()

    cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('input_image',image)
    # cv.waitKey(0)
    cv.namedWindow('image_gray')
    cv.imshow('image_gray', image_gray)
    # cv.waitKey(0)
    cv.namedWindow('image_bit')
    cv.imshow('image_bit', image_bit)
    cv.waitKey(0)
    cv.destroyAllWindows()

