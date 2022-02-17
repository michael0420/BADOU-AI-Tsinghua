# -*- coding: utf-8 -*-
"""
@author: orea

彩色图像的灰度化、二值化
"""

import cv2
import numpy as np


def img2gray(img):
    bgr_coff = np.zeros(img.shape)
    bgr_coff[:, :, 0] = 0.11
    bgr_coff[:, :, 1] = 0.59
    bgr_coff[:, :, 2] = 0.3
    img_gray = img * bgr_coff
    img_gray = np.sum(img_gray, axis=2)
    img_gray = img_gray.astype(np.uint8)
    return img_gray


def img2bin(img_gray, thresh=0.5):
    img_gray = img_gray / 255.0
    img_bin = np.where(img_gray >= thresh, 255, 0)
    img_bin = img_bin.astype(np.uint8)
    return img_bin


img_path = "lenna.png"
img = cv2.imread(img_path)
img_gray = img2gray(img)
img_bin = img2bin(img_gray, 0.5)
cv2.imwrite("img_gray.png", img_gray)
cv2.imwrite("img_bin.png", img_bin)
