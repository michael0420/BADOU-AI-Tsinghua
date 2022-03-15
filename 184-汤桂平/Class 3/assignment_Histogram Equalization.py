# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/2/27

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 图像均衡化函数实现
def imageHistEqualization(image):
    height, width = image.shape
    totalPixel = height * width
    print('the total pixels:', totalPixel)

    temp = {}
    sumValue = 0
    for item in np.sort(image.flatten()):
        temp.setdefault(item,0)
        temp[item] += 1
    for item in temp.keys():
        prob = temp[item]/totalPixel
        sumValue += prob
        temp[item] = sumValue * 255
    for i in range(height):
        for j in range(width):
            image[i][j] = temp[image[i][j]]
    return image

# 灰度直方图均衡化
image = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dst = imageHistEqualization(img_gray)
Hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
cv2.imshow('source gray image and equalized image', np.hstack([img_gray,dst]))
cv2.waitKey(0)

plt.figure()
plt.title('histogram of equalized image')
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(Hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()

'''
# 彩色直方图均衡化
imgB, imgG, imgR = cv2.split(image)
HistimgB = imageHistEqualization(imgB)
HistimgG = imageHistEqualization(imgG)
HistimgR = imageHistEqualization(imgR)
equalizedColorImage = cv2.merge(HistimgB, HistimgG, HistimgR)
plt.figure()
plt.hist(equalizedColorImage.ravel(), 256, [0,255])
plt.title('equalized histogram')
plt.show()

cv2.imshow("equalized color image", equalizedColorImage)
cv2.waitKey(0)
'''