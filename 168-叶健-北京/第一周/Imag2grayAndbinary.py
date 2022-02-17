# -*- coding: utf-8 -*-
"""
彩色图像的灰度化、二值化
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
# 手动灰度化
mag = cv2.imread("lenna.png")
# print(mag.shape[:2])
h, w = mag.shape[:2]
# print(mag[1, 1])
mag_gray = np.zeros([h, w], mag.dtype)
for i in range(h):
    for j in range(w):
        mag_gray[i, j] = int(mag[i, j][0]*0.11+mag[i, j][1]*0.59+mag[i, j][2]*0.3)  #opencv读取图片 默认通道是BGR,需注意
        # mag_gray[i, j] = int(mag[i, j][0]*0.3+mag[i, j][1]*0.59+mag[i, j][2]*0.11)
print("image show gray: %s"%mag_gray)
cv2.imshow("image show gray", mag_gray)
# cv2.waitKey(0)  #等待键盘输入毫秒, 参数是0 则无限等待,要不然图片一闪而过看不清;
"""

#灰度化
plt.subplot(221)
mag = plt.imread("lenna.png")
plt.imshow(mag)
mag2 = cv2.imread("lenna.png")
# cv2.imshow("tu",mag)
# cv2.waitKey(0) #试试看cv2读取的原图效果
plt.subplot(222)
mag_gray = cv2.cvtColor(mag2, cv2.COLOR_BGR2RGB)  #  opencv读取原图BGR转RGB
plt.imshow(mag_gray) #查看图片
mag_gray = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)
print("=====imag1 lenna.png=====")
print(mag)
# mag_gray = rgb2gray(mag)
plt.subplot(223)
plt.imshow(mag_gray,cmap='gray')
print("=====imag2 lenna.png=====")
print(mag_gray)

#二值化
# print(mag_gray.shape)
"""#手动二值化
rows, lines = mag_gray.shape
for i in range(rows):
    for j in range(lines):
        if mag_gray[i, j] > 0.5:
            mag_gray[i, j] = 1
        else:
            mag_gray[i, j] = 0
plt.subplot(224)
plt.imshow(mag_gray, cmap='gray')"""
mag_binary=np.where(mag_gray>0.5,1,0)
plt.subplot(224)
plt.imshow(mag_binary, cmap='gray')
print("=====imag3 lenna.png=====")
print(mag_binary)
plt.show()