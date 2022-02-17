#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/6 23:55 
# ide： PyCharm



import numpy as np


class PCA:
    def __init__(self, X, K):
        """
        :param X: 训练样本矩阵X
        :param K: 降维成K维
        """
        self.X = X
        self.K = K
        self.centralX = [] # 矩阵X的中心化
        self.C = [] # 协方差矩阵 C
        self.U = [] # 样本X的降维转换矩阵 U
        self.Z = [] # 样本X 的降维矩阵 Z

        self.centralX = self._centralized()
        self.C = self._conv()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        '''矩阵X的中心化'''
        print("样本矩阵X: \n", self.X)
        centralX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print("样本集的特征均值：\n", mean)
        centralX = self.X - mean
        print("中心化之后的矩阵：\n", centralX)
        return centralX

    def _conv(self):
        """求样本X的协方差矩阵"""
        #公式 D = 1/m *(XT * X)
        #样本集的样本总数
        ns = np.shape(self.centralX)[0]
        print(ns)
        #协方差矩阵
        C = np.dot(self.centralX.T, self.centralX) / (ns - 1)
        print("协方差矩阵C: \n", C)
        return C

    def _U(self):
        """求X的降维矩阵"""
        # 先求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.C)
        print("特征值：\n", eig_vals)
        print("特征向量：\n", eig_vectors)

        index = np.argsort(-eig_vals)

        #降维矩阵
        UT = [eig_vectors[:, index[i]] for i in range(self.K)]
        print('----------------------------------------------------------')
        print(UT)
        U = np.transpose(UT)
        print("降维后的转换矩阵：\n", U)
        return U

    def _Z(self):
        """降维后的矩阵"""
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X,K)



























