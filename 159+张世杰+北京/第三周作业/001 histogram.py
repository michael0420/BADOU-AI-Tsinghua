# encoding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
读取图像，并进行灰度化
"""
img = cv2.imread('lenna.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
直接用plt绘制直方图
"""
# 直接在PLT画图的过程中绘制直方图；
# plt.figure()
# plt.hist(gray.ravel(), 256)  # 各个参数还不了解
# plt.show()

"""
用caluHist生成直方图：关于hist = cv2.calcHist([gray], [0], None, [500], [0, 1000]) 后两个参数的理解：
500 是将所有的柱状图分成多少分，而[0, 1000]代表的是像素值，也就是0-255，超过此部分的的值，在图像中则为0；这两个参数同时发生在横轴上，其为对应关系，
即：两个参数的起点和终点是相同的。
"""
# hist = cv2.calcHist([gray],[0],None,[256],[0,256])
# plt.figure()#新建一个图像
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")#X轴标签
# plt.ylabel("# of Pixels")#Y轴标签
# plt.plot(hist)
# plt.xlim([0,256])#设置x坐标轴范围
# plt.show()

"""
彩色直方图绘制
"""
chans = cv2.split(img)  # 生成一个列表
print(chans[0])
colors = ('b', 'g', 'r')  # 为plt.plot参数作准备
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])  # 此处应注意：所有参数都应该放在中括号[]内；
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()
