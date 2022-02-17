# """
#
# Kmeans()函数原型：
# retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
#
#     data: 表示聚类数据，最好是np.float32类型的N维点集
#     K: 表示聚类族数
#     bestLabels: 表示输出的整数数组，用于存储每个样本的聚类标签索引
#     criteria: 表示迭代停止的模式选择，这是一个含有三个元素的元组型数，格式为（type，max_iter,epsilon)
#         其中：type有如下模式：
#             cv2.TERM_CRITERIA_EPS:精确度（误差）满足epsilon停止
#             cv2.TERM_CRITERIA_MAX_ITER：迭代超过max_iter停止
#             cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束
#     attempts: 表示重复实验kmeans算法的次数，算法返回产生的最佳结果的标签
#     flags: 表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS; cv2.KMEANS_RANDOM_CENTERS
#     centers:表示集群中心的输出矩阵，每个集群中心为一行数据
#
# """
#
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv.imread('../../../../../BaiduNetdiskDownload/lenna.png', 0)
print(img.shape)
"""
# cv.imshow('aa', img)
# cv.waitKey(0)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, 'gray')

plt.show()

plt显示图像和 cv 显示图像的格式不一样
"""

#高度，宽度
rows, cols = img.shape[:]

#图像二维像素转换为一维
data = img.reshape(rows * cols, 1)
data = np.float32(data)

#停止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv.KMEANS_RANDOM_CENTERS

#K-MEANS聚类，聚类成4类
compactness, labels, centers = cv.kmeans(data, 4, None, criteria, 10, flags)

#生成最终图像
# centers = np.uint8(centers)
# res = centers[labels.flatten()]
dst = labels.reshape((img.shape[0], img.shape[1]))
print("dst", dst)

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()