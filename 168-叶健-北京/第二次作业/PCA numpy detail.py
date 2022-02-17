# -*- coding: utf-8 -*-

import numpy as np

X = np.array([[10, 15, 29, 5],
              [15, 46, 13, 4],
              [23, 21, 30, 6],
              [11, 9, 35, 3],
              [42, 45, 11, 8],
              [9, 48, 5, 9],
              [11, 21, 14, 12],
              [8, 5, 15, 7],
              [11, 12, 21, 7],
              [21, 20, 25, 8]])
#中心化
X=X-X.mean(axis=0)
print(X)
#协方差
X_cov=np.dot(X.T,X)/X.shape[0]
print(X_cov)
#求特征值和特征向量
values,vectors = np.linalg.eig(X_cov)
print(values,'\n',vectors)

K=2 #降维后的维度数
#求降维需要的变换向量
idx = np.argsort(-values)
UT = vectors[:, idx[:K]]
print(UT)
#降维后的向量
T=np.dot(X,UT)
print(T)


