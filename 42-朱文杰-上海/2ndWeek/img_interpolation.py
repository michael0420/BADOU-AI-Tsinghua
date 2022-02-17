# -*- coding: utf-8 -*-
"""
interpolation

Date:2021.11.27
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.color import rgb2gray

test_img = "lenna.png"

def InterpolationImg(imgSource, imgDstH, imgDstW):
    imgSourceH, imgSourceW, channel = imgSource.shape
    if imgSourceH == imgDstH and imgSourceW == imgDstW:
        return imgSource.copy()
    #rate
    rate_y = imgSourceH / imgDstH
    rate_x = imgSourceW / imgDstW
    #empty dest array
    imgDst = np.zeros((imgDstH, imgDstW, 3), img_source.dtype)
    for i in range(3):
        for dst_y in range(imgDstH):
            for dst_x in range(imgDstW):
                # step one
                src_x = (dst_x + 0.5) * rate_x - 0.5
                src_y = (dst_y + 0.5) * rate_y - 0.5
                # step two
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, imgSourceW - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, imgSourceH - 1)
                # step three
                temp0 = (src_x1 - src_x) * imgSource[src_y0, src_x0, i] + (src_x - src_x0) * imgSource[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * imgSource[src_y1, src_x0, i] + (src_x - src_x0) * imgSource[src_y1, src_x1, i]
                # step four
                imgDst[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return imgDst


if __name__ == '__main__':
    img_source = cv2.imread(test_img)
    image_interpolation=InterpolationImg(img_source, 800, 800)
    cv2.imshow('img_source', img_source)
    cv2.imshow('image_interpolation', image_interpolation)
    cv2.waitKey(0)