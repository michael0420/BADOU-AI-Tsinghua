#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt
#加载原图片
img = cv2.imread("lenna.png", 1)
# 把图像转换成灰度图像再进行直方图均衡化
length=img.shape[0]
width=img.shape[1]
src_gray = np.zeros([length,width],img.dtype)
for i in range(length):
    for j in range(width):
        #浮点算法
        src_gray[i][j]=img[i][j][0]*0.11+img[i][j][1]*0.59+img[i][j][2]*0.3
dst_gray = cv2.equalizeHist(src_gray)

# 直方图
hist = cv2.calcHist([dst_gray],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst_gray.ravel(), 256)
plt.show()

cv2.imshow("src_gray",src_gray)
cv2.imshow("dst_gray",dst_gray)
#cv2.waitKey(0)

# 彩色图像直方图均衡化
cv2.imshow("src_rgb", img)
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
src_b = cv2.equalizeHist(b)
src_g = cv2.equalizeHist(g)
src_r = cv2.equalizeHist(r)
# 合并三个通道
result = cv2.merge((src_b, src_g, src_r))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)

