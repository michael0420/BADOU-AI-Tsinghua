#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双线性插值"""
import cv2 as cv
import numpy as np

def zoom_image(image,dst_width,dst_height):
    src_width = image.shape[0] -1
    src_height = image.shape[1] -1
    new_image = np.zeros([dst_width, dst_height, image.shape[2]], np.uint8)
    for dst_i in range(dst_width):
        for dst_j in range(dst_height):
            print("dst_i=:%d,dst_j=:%d"%(dst_i,dst_j))
            src_i = max(((dst_i+0.5)*src_width/dst_width)-0.5,0)
            src_j = max(((dst_j+0.5)*src_height/dst_height)-0.5,0)
            FQ11 = image[int(src_i),int(src_j)]
            FQ21 = image[int(src_i)+1,int(src_j)]
            FQ12 = image[int(src_i), int(src_j)+1]
            FQ22 = image[int(src_i)+1, int(src_j) + 1]
            # (y2-y)((x2-x)FQ11+(x-x1)FQ21)+(y-y1)((x2-x)FQ12+(x-X1)FQ22)
            x1=int(src_i)
            x2=int(src_i)+1
            x=src_i
            y1=int(src_j)
            y2=int(src_j)+1
            y=src_j
            print("src_i=:%d,src_j=:%d" % (src_i, src_j))
            new_image[dst_i,dst_j] = (y2-y)*((x2-x)*FQ11+(x-x1)*FQ21)+(y-y1)*((x2-x)*FQ12+(x-x1)*FQ22)

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