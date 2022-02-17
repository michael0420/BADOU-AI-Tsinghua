# author: woo

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('kmeans.jpeg') 

# 图像二维像素转换为一维
data = img.reshape((-1,3))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means
compactness, labels, centers2 = cv2.kmeans(data, 5, None, criteria, 10, flags)
labels = labels.reshape((img.shape[0], img.shape[1]))
print(np.unique(labels), labels.shape)

label_show = np.zeros(img.shape)
label_show[labels == 1] = [255, 255, 255]
label_show[labels == 2] = [0, 255, 0]
label_show[labels == 3] = [0, 0, 255]
label_show[labels == 4] = [255, 0, 0]
cv2.imwrite("kmeans.png", label_show)
