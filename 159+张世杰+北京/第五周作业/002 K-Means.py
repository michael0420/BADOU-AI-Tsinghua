# encoding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 0)  # 读取到灰度图像
rows, cols = img.shape[:]
'''将图像转换为一纬，改变数据格式为浮点型'''
data = img.reshape((rows * cols, 1))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 停止条件为达到精度或达到迭代次数，精度为1，最大迭代次数10
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)  # 生成的三个参数：紧密度、标签、中心

dst = labels.reshape((img.shape[0], img.shape[1]))
print(dst)

plt.imshow(dst,'gray')
plt.show()
