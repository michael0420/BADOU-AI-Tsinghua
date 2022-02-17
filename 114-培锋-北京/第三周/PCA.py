# -*- coding: utf-8 -*-
"""
主成分分析
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):#构造方法
        '''
        :param X,训练样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得


    def _centralized(self):
        '''第一步，矩阵中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值，每一维度的平均数
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  ##样本集的中心化，每一个维度的数据减去同维度的平均值
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''第二步，求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]#求出数组的维度，也就是列数
        print(ns)

        #每列都是一个维度，X,Y,Z
        #   COV(X,X)  COV(X,Y)  COV(X,Z)
        #   COV(Y,X)  COV(Y,Y)  COV(Y,Z)
        #   COV(Z,X)  COV(Z,Y)  COV(Z,Z)
        #
        #   cov(X,X) = 求和(X1*X1 + X2*X2 + ... +Xn*Xn)
        #   cov(X,Y) = 求和(X1*Y1 + X2*Y2 + ... +Xn*Yn)
        #   ...
        #   cov(Z,Y) = 求和(Z1*Y1 + Z2*Y2 + ... +Zn*Yn)
        #   cov(Z,Z) = 求和(Z1*Z1 + Z2*Z2 + ... +Zn*Zn)
        # 样本矩阵的协方差矩阵C。求协方差矩阵
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''求协方差矩阵的特征值、特征向量'''
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        #w, v = numpy.linalg.eig(a)，a为方阵
        #计算方形矩阵a的特征值和右特征向量
        a, b = np.linalg.eig(self.C)  # 特征值赋值给a，对应特征向量赋值给b
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        #返回值为原始数组排序后，对应到原始数组的下标，索引，
        #例:[335.15738485  95.32771231  32.63712506]
        #返回[0,1,2]
        #若为升序排序，返回[2,1,0]
        ind = np.argsort(-1 * a)#原本是升序排序，乘-1，就可变为降序排序
        print(ind)
        print(a)
        # 构建K阶降维的降维转换矩阵U
        #按照特征值由大到小的顺序，将对应的特征向量进行组合
        UT = [b[:, ind[i]] for i in range(self.K)]
        print("UT",UT)
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    #定义一二位数组，也就是矩阵
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
    #
    K = np.shape(X)[1] - 1
    L = np.shape(X)#(10，3)打印出矩阵的行数：10，列数：3
    print(L)
    print(K)
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)#X代表矩阵,k为列数-1,降维后的值。k阶降维矩阵



