#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/5 17:27 
# ide： PyCharm


'''
直方图均衡化
equalHist(src, dst=None)
src: 单通道图像
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../../../../BaiduNetdiskDownload/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#均衡化
dst = cv2.equalizeHist(gray)
# cv2.imshow("1", dst)
# cv2.waitKey(0)

#均衡化前后的直方图对比
# plt.figure()
# plt.hist(gray.ravel(), 256)
# plt.hist(dst.ravel(), 256)
# plt.show()

#灰度图和均衡化图对比
# cv2.imshow("after", np.hstack([gray, dst]))
# cv2.waitKey(0)



#彩色直方图均衡化
img = cv2.imread("../../../../BaiduNetdiskDownload/lenna.png", 1)
# cv2.imshow("origin", img)
# cv2.waitKey(0)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

#合并通道
res = cv2.merge((bH, gH,rH))
cv2.imshow('color', res)
cv2.waitKey(0)











