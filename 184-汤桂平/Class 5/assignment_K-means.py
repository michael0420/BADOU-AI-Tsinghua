# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/3/26

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

img = cv2.imread('lenna.png', 0)   # 把图像读取成灰度数据
img = np.float32(img)
data = img.flatten()    # 需要聚类的数据，最好是np.float32的数据，每个特征放一列
print('data to be clustered\n', data)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 设置迭代停止的条件
flags = cv2.KMEANS_RANDOM_CENTERS  # 设置初始中心的选择模式

'''
kmeans聚类函数中的compactness返回的是每个点到相应重心的距离的平方和，为一个浮点数；
labels返回的是data中每一个数据元素被聚类后的类标签，即序号0，1，2，……，为一个data.shape[0]行1列的二维数组；
centers返回的是聚类的中心组成的数组，即K行1列的二维数组；
'''
compactness, labels, centers = cv2.kmeans(data, 5, None, criteria, 10, flags)

print('compactness:', compactness)
print('labels\n', labels)
print('centers\n', centers)
img_kmeans = labels.reshape((img.shape[0], img.shape[1]))

plt.rcParams['font.sans-serif']=['SimHei']   # 用来正常显示中文标签

titles = [u'原始图像', u'聚类图像']
images = [img, img_kmeans]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
