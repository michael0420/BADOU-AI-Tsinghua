#!/usr/bin/env python 
# coding:utf-8
import numpy as np


class PCA():
    def __init__(self, component):
        self.component = component

    # 中心化
    def center_matrix(self, X):
        # axis=0 对各列求均值
        X = X - np.mean(X, axis=0)
        return X

    # 求协方差矩阵
    def covariance_trans(self, X):
        # 样本数量
        sample = X.shape[0]
        # 协方差矩阵
        cov = np.dot(X.T, X) / (sample - 1)
        return cov

    # 求特征值与特征向量
    def get_feature(self, cov):
        # eigenvalues是特征值,vector为特征向量
        eigenvalues, vector = np.linalg.eig(cov)
        # 将特征值倒序排序
        index = np.argsort(-eigenvalues)
        # 获取特征向量矩阵
        w = vector[:, index[:self.component]]
        return w

    def fit_transform(self, X):
        X = self.center_matrix(X)
        cov = self.covariance_trans(X)
        w = self.get_feature(cov)
        # 对数据进行降维
        result = np.dot(X, w)
        return result


if __name__ == '__main__':
    # 10样本3特征的样本集,行为样例,列为特征维度
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])

    # 降维到2阶矩阵
    pca = PCA(component=2)
    # PCA训练
    result = pca.fit_transform(X)
    print("降维后的矩阵为:\n", result)
