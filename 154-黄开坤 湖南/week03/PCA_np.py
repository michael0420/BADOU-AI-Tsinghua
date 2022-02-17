#coding:utf-8

import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]       #单个末尾下划线（后缀）是一个约定，用来避免与Python关键字产生命名冲突.
        #中心化并求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / (X.shape[0] - 1)
        #求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        #获得降为排列特征值额索引
        ind_k = np.argsort(-eig_vals)   #从大到小
        #降为矩阵
        self.components_ = eig_vectors[:, ind_k[: self.n_components]]
        #对X进行降维
        new_X = np.dot(X, self.components_)
        return new_X

#调用类：
pca = PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
print('X的样本数n和特征数k：', X.shape)
new_X = pca.fit_transform(X)
print('输出降维后的数据为：\n', new_X)
print('--------------------------------')


'''调用sklearn库中的PCA'''

from sklearn.decomposition import PCA

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
pca = PCA(n_components=2)       #降到2维
pca.fit(X)                      #训练
# print(pca.fit(X))     #PCA(n_components=2)
newX = pca.fit_transform(X)     #降维后的数据
print('贡献率：', pca.explained_variance_ratio_)    #输出贡献率
print(newX)
