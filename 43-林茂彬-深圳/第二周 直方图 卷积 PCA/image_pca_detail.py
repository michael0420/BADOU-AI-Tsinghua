# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
https://blog.csdn.net/daaikuaichuan/article/details/53444639
https://blog.csdn.net/duanyule_cqu/article/details/54959897
https://blog.csdn.net/yangleo1987/article/details/52845912
"""

import numpy as np


class PCA:
    def __init__(self, X, K):#初始化实例属性
        self.X = X       #样本矩阵X
        self.K = K       #K阶降维矩阵的K值
        self.center=[]        #创建存放中心化后的数据
        self.center=self._center()
        self.cov = []
        self.cov=self._cov()
        self.U = []
        self.U = self._eigen()
        self.Z = []
        self.Z = self._Z()

    #1、去除平均值，让每一维特征减去各自特征的平均值
    def _center(self):#求出均值
        mean = np.mean(self.X,0)#样本集的特征均值 np.mean(now2,0) # 压缩行，对各列求均值   np.mean(now2,1) # 压缩列，对各行求均值
        print(mean)
        self.centrX = []
        self.centrX = self.X - mean ##样本集的中心化
        print('去中心化数据%s'%self.centrX )



        #2、先让样本矩阵中心化，即每一维度减去该维度的均值，使每一维度上的均值为0，然后直接用新的到的样本矩阵的转置乘上新的到的样本，然后除以(N-1)即可。其实这种方法也是由前面的公式通道而来，只不过理解起来不是很直观，但在抽象的公式推导时还是很常用的！
    #理解协方差矩阵的关键就在于牢记它计算的是不同维度之间的协方差，而不是不同样本之间，拿到一个样本矩阵，我们最先要明确的就是一行是一个样本还是一个维度。


    def _cov(self):
        '''求样本矩阵X的协方差矩阵'''
        #样本集的样例总数
        ns = np.shape(self.centrX)[0]
        print('样本集的样例总数:\n', ns)
        #样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1) #矩阵乘法np.dot    中心化矩阵的协方差矩阵公式1/m * x.T * x
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    #3、 求X的协方差矩阵C的特征值和特征向量    特征向量：就是求旋转角度,求新坐标轴方向  特征值 新坐标轴下的反差
    # 对数字图像矩阵做特征值分解，其实是在提取这个图像中的特征，这些提取出来的特征是一个个的向量，即对应着特征向量。而这些特征在图像中到底有多重要，这个重要性则通过特征值来表示。
    # https://www.bilibili.com/video/BV1E5411E71z


    def _eigen(self):
        eigen,fea_vector=np.linalg.eig(self.cov)
        print('样本集的协方差矩阵C的特征值:\n', eigen)
        print('样本集的协方差矩阵C的特征向量:\n', fea_vector)

        #给出特征值降序的topK的索引序列
        ind = np.argsort(-1*eigen)
        print('ind:',ind)
        #构建K阶降维的降维转换矩阵U
        UT = [fea_vector[:,ind[i]] for i in range(self.K)]   #因为计算下来的是不同纬度下的特征向量，从大到小排好后直接取主要成分，也就是自己设定的k
        print(UT)
        #将特征值按照从大到小的排序，选择其中最大K个，然后将其对应的K个特征向量分别作为列向量组成特征向量矩阵
        U = np.transpose(UT)

        print('%d阶降维转换矩阵U:\n'%self.K, U)
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
    #print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)
