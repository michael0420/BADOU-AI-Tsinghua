# encoding: utf-8
import numpy as np


class CPCA(object):
    # 对类进行初始化，传入参数：数据集矩阵，所需要的降纬后的维度
    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.C = []  # 样本集的协方差矩阵C
        self.centrX = []
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()  # 调用中心化函数 , 私有函数不可以直接用类名调用， 得用self
        self.C = self._cov()  # 调用协方差矩阵
        self.U = self._U()  # 获取特征值，并排序
        self.Z = self._Z()  # 转换向量空间后的数据集

    # 中心化（去均值化）
    def _centralized(self):  # 私有函数
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 获取样本的均值， 此以行为单位进行取均值，所以需要转置
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  # 中心化
        print('去均值化后的数据集为:\n', centrX)
        return centrX

    # 获取样本的协方差矩阵, 此处根据协方差的计算公式，在中心化后，协方差的公示可化简为矩阵乘法，此处根据数理统计对协方差的定义除以：ns-1
    def _cov(self):
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    # 求协方差矩阵的特征值和特征向量：此处通过特征向量对数据进行处理后，生成的在新的向量空间的数据信息的协方差为0
    # 这块的 np 的公式需要重点学习一下
    def _U(self):
        a, b = np.linalg.eig(self.C)  # 直接通过np函数获得矩阵的特征值和特征向量, 此处生成的特征向量就是以列显示的
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)  # 为特征值设置索引不理解NP的这个公式
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    # 对原来数据集进行降维处理：在这里一定要注意行列式的行和列的意义
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


# 函数的调用：
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
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)
