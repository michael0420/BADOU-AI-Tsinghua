# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

#
# # 原图操作 - - - - - ok
start_img = cv2.imread("lenna.png")    # 读取photo
print(start_img.shape)    # 三元组代表图片的 high weight deep

# print("2 d array of start_img:")
# print(start_img)  # 512 * 512 * 3

cv2.imshow("start_img", start_img)    # 显示
cv2.waitKey(0)

# 画布
plt.subplot(221)
plt.imshow(plt.imread("lenna.png"))
00


# 灰度图操作 - - - - -
# 1 利用公式
# 获取宽和高
h, w = start_img.shape[0:2]
print(h, w)

img_gray_1 = np.zeros([h, w], start_img.dtype)
for i in range(h):
    for j in range(w):
        color = start_img[i, j]    # 取出BGR坐标中的光学三原色
        img_gray_1[i, j] = color[0] * 0.11 + color[1] * 0.59 + color[2] * 0.3    # RGB

cv2.imshow("img_gray_1", img_gray_1)    # 显示
cv2.waitKey(0)
print(img_gray_1)

# 画布
plt.subplot(222)
plt.imshow(img_gray_1, cmap='gray')


# 2 调库
img_gray_2 = cv2.cvtColor(start_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray_2", img_gray_2)
cv2.waitKey(0)

# 画布
plt.subplot(223)
plt.imshow(img_gray_2, cmap='gray')


# 二值化
img_rgb = plt.imread("lenna.png")    # 读取图片
gray_to_bin = rgb2gray(img_rgb)    # 转为灰度图
print(gray_to_bin.shape)
h, w = gray_to_bin.shape
print(gray_to_bin)
# gray_to_bin = np.where(gray_to_bin >= 0.5, 1, 0)
for i in range(h):    # 二值化过程 遍历
    for j in range(w):
        if gray_to_bin[i, j] < 0.45:
            gray_to_bin[i, j] = 0
        else:
            gray_to_bin[i, j] = 10
#cv2.imshow("bin of img", a)    no
#cv2.waitKey(0)

# 画布
plt.subplot(224)
plt.imshow(gray_to_bin, cmap='gray')
print(gray_to_bin)

# 显示画布
plt.show()
