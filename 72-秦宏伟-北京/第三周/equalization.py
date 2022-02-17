#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直方图均衡化"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
直方图均衡化
"""
def balance_gray(image):
    if len(image.shape)!=2 :
        return
    src_width = image.shape[0] -1
    src_height = image.shape[1] -1
    new_image = np.zeros([src_width, src_height], np.uint8)

    Ni = np.zeros([256])
    Pi = np.zeros([256])
    sumPi = np.zeros([256])
    sumResult = np.zeros([256])
    #计算Ni
    for src_i in range(src_width):
        for src_j in range(src_height):
            Ni[image[src_i,src_j]]+=1
    #计算Pi，Pi=Ni/image
    for i in range(256):
        Pi[i] = Ni[i]/(src_width*src_height)
    #sumPi
    sum = 0
    for i in range(256):
        if Pi[i] >0:
            sum += Pi[i]
            sumPi[i] = sum
        else:
            sumPi[i] = 0
    for i in range(256):
        #sumPi*256-1
        sumResult[i] = int(sumPi[i]*256-1)

    for src_i in range(src_width):
        for src_j in range(src_height):
            new_image[src_i,src_j] = sumResult[image[src_i,src_j]]
    return new_image

def balance_image(image):
    src_width = image.shape[0] -1
    src_height = image.shape[1] -1
    channel = image.shape[2]
    new_image = np.zeros([src_width, src_height, channel], np.uint8)

    for c_i in range(channel):
        Ni = np.zeros([256])
        Pi = np.zeros([256])
        sumPi = np.zeros([256])
        sumResult = np.zeros([256])
        #计算Ni
        for src_i in range(src_width):
            for src_j in range(src_height):
                Ni[image[src_i,src_j,c_i]]+=1
        #计算Pi，Pi=Ni/image
        for i in range(256):
            Pi[i] = Ni[i]/(src_width*src_height)
        #sumPi
        sum = 0
        for i in range(256):
            if Pi[i] >0:
                sum += Pi[i]
                sumPi[i] = sum
            else:
                sumPi[i] = 0
        for i in range(256):
            #sumPi*256-1
            sumResult[i] = int(sumPi[i]*256-1)

        for src_i in range(src_width):
            for src_j in range(src_height):
                new_image[src_i,src_j,c_i] = sumResult[image[src_i,src_j,c_i]]

    return new_image
if __name__ == '__main__':
    image = cv.imread('lenna.png')
    (b, g, r) = cv.split(image)
    image_b = balance_gray(b)
    image_g = balance_gray(g)
    image_r = balance_gray(r)
    new_image = cv.merge((image_b,image_g,image_r))

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    new_gray = balance_gray(gray)


    image_RGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)

    new_image_RGB = cv.cvtColor(new_image,cv.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(image_RGB)
    plt.subplot(2, 2, 2)
    plt.imshow(new_image_RGB)
    plt.subplot(2, 2, 3)
    plt.imshow(gray,cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(new_gray,cmap='gray')
    plt.show()

    plt.figure()
    plt.hist(gray.ravel(), 256)
    plt.show()
    plt.figure()
    plt.hist(new_gray.ravel(), 256)
    plt.show()

    cv.namedWindow('src_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('src_image', image)
    cv.namedWindow('new_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('new_image',new_image)

    cv.namedWindow('src_gray', cv.WINDOW_AUTOSIZE)
    cv.imshow('src_gray', gray)
    cv.namedWindow('new_gray', cv.WINDOW_AUTOSIZE)
    cv.imshow('new_gray',new_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()



