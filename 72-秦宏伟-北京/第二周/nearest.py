#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最近邻"""
import cv2 as cv
import numpy as np

def zoom_image(image,dst_width,dst_height):
    src_width = image.shape[0] -1
    src_height = image.shape[1] -1
    new_image = np.zeros([dst_width, dst_height, image.shape[2]], np.uint8)
    for dst_i in range(dst_width):
        for dst_j in range(dst_height):
            src_i = int(round(dst_i*src_width/dst_width))
            src_j = int(round(dst_j*src_height/dst_height))
            new_image[dst_i,dst_j] = image[src_i,src_j]
    return new_image
if __name__ == '__main__':
    image = cv.imread('lenna.png')
    new_image = zoom_image(image,800,800)

    new_image2 = zoom_image(image, 200, 200)

    cv.namedWindow('src_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('src_image', image)
    cv.namedWindow('new_image', cv.WINDOW_AUTOSIZE)
    cv.imshow('new_image',new_image)
    cv.namedWindow('new_image2', cv.WINDOW_AUTOSIZE)
    cv.imshow('new_image2', new_image2)
    cv.waitKey(0)
    cv.destroyAllWindows()