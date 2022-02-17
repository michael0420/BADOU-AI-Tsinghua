#!/user/bin/env python
# encoding=gbk
"""
@author: 陈志海
@pca：计算pca：
    step1：计算样本x各维度均值，及其去中心化Z = x - x.mean
    step2：计算去中心化矩阵的协方差矩阵cov = Z.T * Z / m。m为x的行
    step3：计算协方差矩阵的特征值和特征向量矩阵eigen_Value和eigen_vector
    step4：提取最大的前n个特征值对应的特征向量矩阵
    step5：样本x与提取的特征向量矩阵相乘，得pca降维后的结果
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._base import load_iris

def pca(x, n_feature=2):
    x = x - x.mean(axis=0)
    cov = np.dot(x.T, x) / x.shape[0]
    eigen_value, eigen_vector = np.linalg.eig(cov)
    idx_desc = np.argsort(-eigen_value)
    vector_desc = eigen_vector[:, idx_desc[: n_feature]]
    result = np.dot(x, vector_desc)
    return result


# main
# 导入数据，维度为4
x = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], \
              [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
result = pca(x, 2)
print("pca of x:")
print(result)

# 鸢尾花
x, y = load_iris(return_X_y=True)
result = pca(x, 2)
plt.figure("鸢尾花, pca n_feature=2")
for i in range(y.shape[0]):
    if y[i] == 0:
        c = 'r'
        marker = 'x'
    elif y[i] == 1:
        c = 'b'
        marker = 'D'
    else:
        c = 'g'
        marker = '.'
    plt.scatter(x[i, 0], x[i, 1], color=c, marker=marker)
plt.show()
