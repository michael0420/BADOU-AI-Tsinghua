#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/11/24 22:19
# ide： PyCharm



from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("../../../BaiduNetdiskDownload/lenna.png")
h, w = img.shape[:2]
print(img.shape)
print(img)

img_gray = np.zeros((h, w), img.dtype)

# print(img[0][40:45])
# print(img[1][40:45])
# cv2 是按照bgr方式读取的
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
# print(img_gray)
cv2.imshow('origin_img', img)
cv2.imshow('img_gray', img_gray)
cv2.waitKey(0)

#未灰度化
plt.subplot(221)
img = plt.imread("../../../BaiduNetdiskDownload/lenna.png")
plt.imshow(img)
# print("image lenna")
# print(img)


#灰度化
img_gray = rgb2gray(img)
# img = cv2.imread('../../../BaiduNetdiskDownload/lenna.png')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')


#二值化
#第一种方式
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i][j] <= 0.5:
            img_gray[i][j] = 0
        else:
            img_gray[i][j] = 1

plt.subplot(223)
plt.imshow(img_gray, cmap='gray')

#第二种方式
img_gray = rgb2gray(img)
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(224)
plt.imshow(img_binary, cmap='gray')
plt.show()


