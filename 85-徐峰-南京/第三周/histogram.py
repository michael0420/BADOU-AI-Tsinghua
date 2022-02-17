#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/5 0:00 
# ide： PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt


# img = cv2.imread("../../../../BaiduNetdiskDownload/lenna.png", flags=0)#里面的参数 flags = 1 (彩色）, 0(灰度图）
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.imshow("gray", gray)
# cv2.imshow("color", img)
# cv2.waitKey(0)


#灰度图的直方图
# img = cv2.imread("../../../../BaiduNetdiskDownload/lenna.png", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.figure()
# plt.hist(gray.ravel(), 256)
# plt.show()



# img = cv2.imread("../../../../BaiduNetdiskDownload/lenna.png", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.figure()
# plt.title("histogram")
# plt.xlabel('bins') #
# plt.ylabel('# of pixes') #y
# plt.plot(hist)
# plt.xlim([0, 256]) #坐标轴范围
# plt.show()

#彩色直方图

img = cv2.imread('../../../../BaiduNetdiskDownload/lenna.png', 1)
chanels = cv2.split(img)
print(chanels)
colors = ('b', 'g', 'r')
plt.figure()
plt.title("color histogram")
plt.xlabel('bins')
plt.ylabel('# of pixels')

for (chan, color) in zip(chanels, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)# color = b ,g r
    plt.xlim([0, 256])
plt.show()
