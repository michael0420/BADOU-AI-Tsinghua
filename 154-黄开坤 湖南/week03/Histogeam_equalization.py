#coding:utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt


'''直方图均衡化'''
img = cv2.imread('lenna.png',1) #第二参数。cv2.IMREAD_COLOR : 默认使用该种标识。加载一张彩色图片，忽视它的透明度。cv2.IMREAD_GRAYSCALE : 加载一张灰度图。cv2.IMREAD_UNCHANGED : 加载图像，包括它的Alpha通道。 友情链接：Alpha通道的概念提示：如果觉得以上标识太麻烦，可以简单的使用1，0，-1代替。（必须是整数类型）
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_equal = cv2.equalizeHist(img_gray)    #灰度图像直方图均衡化

plt.figure()
plt.hist(img_equal.ravel(), 256)
plt.show()

cv2.imshow('Histogram Equalization', np.hstack([img_gray, img_equal]))
cv2.waitKey(0)

'''彩色图均衡化'''
# img = cv2.imread('lenna.png')
# (b, g, r) = cv2.split(img)
# bE = cv2.equalizeHist(b)
# gE = cv2.equalizeHist(g)
# rE = cv2.equalizeHist(r)
# #合并通道
# total = cv2.merge((bE, gE, rE))
# cv2.imshow('Color Equalization', total)
# cv2.waitKey(0)


'''自定义实现'''
# img = cv2.imread('lenna.png')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# #step1:获得直方图每个灰度级出现的频数
# #方式1
# hist = np.zeros((256,1))
# # print(hist.shape, hist)
# for x in range(img_gray.shape[0]):
#     for y in range(img_gray.shape[1]):
#         # print(img_gray[x][y],img_gray[x, y])    # 一样是像数值
#         hist[img_gray[x, y]] = hist[img_gray[x, y]] + 1
# # print(hist, hist.shape)
# #方式2# hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# # # print(hist, hist.shape)
#
# #step2累计归一化直方图
# pi = np.zeros(256)
# hig, wid = img_gray.shape[0], img_gray.shape[1]
# for i in range(hist.shape[0]):
#     pi[i] = hist[i] / (hig * wid)   # Pi=Ni/img.size 0~1
# # 累加求和sum(pi)
# sum_pi = np.zeros(256)
# for i in range(1, 256):
#     sum_pi[0] = pi[0]
#     sum_pi[i] = sum_pi[i-1] + pi[i]     #sum(Pi)
#
# #step3: 计算新像素的值
# img_new = np.zeros((hig, wid),dtype=np.uint8)
# for x in range(wid):
#     for y in range(hig):
#         img_new[x, y] = int(sum_pi[img_gray[x, y]] * 256 -1)    #sum_pi(gray)*256-1
# print(img_new, img_new.size)
#
# #显示均衡化后的直方图
# plt.figure()
# plt.title('histogram 2 equalization')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.hist(img_new.ravel(), 256)
# plt.show()
# #显示两个图像对比
# cv2.imshow('Histogram Equalization', np.hstack([img_gray, img_new])) #np.hstack():在水平方向上平铺.函数原型：numpy.hstack(tup)其中tup是arrays序列
# cv2.waitKey(0)
