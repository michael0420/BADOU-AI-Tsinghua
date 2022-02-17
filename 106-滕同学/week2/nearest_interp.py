# -*- coding: utf-8 -*-
"""
@author: orea

彩色图像的最近邻插值
"""

import cv2
import numpy as np


def nearest_image(image, shape):
    new_h, new_c = shape
    h, w, c = image.shape
    new_image = np.zeros((new_h, new_c, c), np.uint8)
    sh = new_h / h
    sw = new_c / w
    for i in range(new_h):
        for j in range(new_c):
            x = int(i / sh)
            y = int(j / sw)
            new_image[i, j] = img[x, y]

    return new_image


img = cv2.imread("lenna.png")
img_resize = nearest_image(img, (800, 800))
cv2.imwrite("nearest_lenna.png", img_resize)
