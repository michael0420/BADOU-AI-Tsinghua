# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/03/10

import numpy as np

class PCA(object):
    # 构造函数
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centrX = []   #为什么矩阵要提前定义，而有些变量不需要提前定义
        self.C = []
        self.U = []
        self.M = []

        self.centrX = self.centralized()
        self.C = self.cov()
        self.U = self._U()
        self.M = self._M()

    # 中心化函数centrX=X-mean,其中mean为各维度样本均值，为列表类型；即centrX是X各维度的样本值分别减去同维度均值得到的
    def centralized(self):
       mean = np.array([np.mean(col) for col in self.X.T])  # 当col分别为X的各列向量时，然后求其均值得到的一维向量赋给mean
       print('矩阵X各列均值：', mean)
       centrX = []
       centrX = self.X - mean
       print('X的中心化矩阵centrX:', centrX)
       return centrX

    # 求协方差矩阵 U = dot(self.centrX.T, self.centrX)/(m-1)  m为样本个数
    def cov(self):
       m = np.shape(self.centrX)[0]
       print('X的样本个数:', m)
       C = np.dot(self.centrX.T, self.centrX)/(m-1)
       print('X的协方差矩阵C:', C)
       return C

    '''求转换矩阵U，先利用np库函数linalg.eig()求出协方差矩阵C的特征值和对应的特征向量，
       然后对特征值从大到小进行排序，然后取对应的特征向量按照特征值大小顺序排列成矩阵U
    '''
    def _U(self):
        a, b = np.linalg.eig(self.C)
        print('协方差矩阵的所有特征值a：', a)
        print('协方差矩阵的所有特征向量：', b)
        ind = np.argsort(-1*a)  #把特征值从大到小排序，并返回其索引
        # U = []
        UT = np.array(b[:,ind[i]] for i in range(self.K))
        U = np.transpose(UT)
        print('转换矩阵U:', U)
        return U

    # 求降维矩阵M=XU
    def _M(self):
        # M = []
        M = np.dot(self.X, self.U)
        # a, b = np.shape(M)
        print('降维矩阵M：', M)
        print('降维矩阵M的行数和列数', np.shape(M))
        return M

if __name__ == '__main__':

    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    K = np.shape(X)[1] - 1
    print('样本降维后的维数', K)
    pca = PCA(X, K)











