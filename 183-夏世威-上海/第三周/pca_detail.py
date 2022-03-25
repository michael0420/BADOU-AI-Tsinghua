"""
不管是特征选择还是特征提取,主要目的都是完成样本数据的降维
pca
目的:
1.同一维度方差最大
2.不同维度相关性为0
方法:
通过计算协方差矩阵的特征值与特征向量实现降维
"""

import numpy as np

sample = np.random.randint(0, 255, size=[20, 4])
K = np.shape(sample)[1] - 2  # 4个特征 -> 2个特征

# 通过X.T 来计算每个维度的均值
mean = np.array([np.mean(attr) for attr in sample.T])

# 样本数据中心化
X = sample - mean

# 计算协方差矩阵,因为中心化,协方差矩阵= (X*X.T)/样本数
X = np.dot(X.T, X) / np.shape(sample)[0]

# 计算特征值与特征向量
values, tensors = np.linalg.eig(X)
print("特征值:\n", values)
print("特征向量:\n", tensors)

# 按照特征值从大到小排列
indexs = np.argsort(values * -1)
feature_result = []
for i in range(K):
    feature_result.append(tensors[indexs[i]])
feature_result = np.transpose(feature_result)
print("特征矩阵:\n", feature_result)

# 计算降维后的数据
result = np.dot(sample, feature_result)
print("pca降维结果:\n", result)
