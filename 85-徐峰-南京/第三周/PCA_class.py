#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/6 23:03 
# ide： PyCharm


import numpy as np

class PCA:
    def __init__(self, n_components): # n_components 指的是降的目标维度
        self.n_components = n_components

    def fit_transform(self, X):
        #特征值
        self.n_features = X.shape[0]

        #求协方差矩阵
        X = X - X.mean(axis=0)
        #协方差公式D= 1/m *(ZT * Z)
        self.covariance = np.dot(X.T, X) / X.shape[0]

        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)

        # 获得排序特征值的序号
        index = np.argsort(-eig_vals)

        #降维矩阵
        self.componets_ = eig_vectors[:, index[:self.n_components]]
        print('-------------------------------:\n', eig_vectors)

        print(self.componets_)
        print('-------------------------------')

        #对X进行降维
        return np.dot(X, self.componets_)

#test
pca = PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)