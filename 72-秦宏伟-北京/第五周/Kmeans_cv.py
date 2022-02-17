# coding: utf-8

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

import cv2 as cv
# import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread('lenna.png',0)
x_shape,y_shape = image.shape

image_k = image.reshape(x_shape*y_shape,1)
image_k = np.float32(image_k)
K = 4
#停止条件 (type,max_iter,epsilon)
criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#设置标签
flags = cv.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv.kmeans(image_k,K,None,criteria,10,flags)

labels=labels.reshape([x_shape,y_shape])

plt.imshow(labels,cmap='gray')
plt.axis('off')
plt.show()
