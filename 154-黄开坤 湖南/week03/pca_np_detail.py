#coding:utf-8

import numpy as np

'''
使用PCA求样本矩阵X的k阶降为矩阵Z
'''


class CPCA(object):
    def __init__(self, X, k):
        '''
        :param X: 训练样本矩阵
        :param k: X的降为矩阵的阶数。即X要特诊要降为成k阶
        '''
        self.X = X      #样本矩阵X
        self.k = k      #k阶降维矩阵的k值
        self.centrX = []#矩阵X的中心化矩阵
        self.Cov = []   #样本集的协方差矩阵
        self.U = []     #样本矩阵X的降维转换矩阵
        self.Z = []     #样本矩阵X的降为后的矩阵

        self.centrX = self._centralized()
        self.Cov = self._Cov()
        self.U = self._U()
        self.Z = self._Z()


    def _centralized(self):
        '''
        :return: 矩阵中心化
        '''
        print('样本矩阵X:\n', self.X)
        # mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值 ##遍历了三行 #[16.1 24.2 19.8]
        mean = np.mean(self.X, axis=0)      #axis=0压缩行，在求均值
        print('样本集特征的均值：\n', mean)
        centrX = self.X - mean              #样本集的中心化，二维减一维
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX


    def _Cov(self):
        '''
        :return: 样本矩阵X的协方差矩阵C
        '''
        #样本集的样本总数
        n = np.shape(self.centrX)[0]
        #计算矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX)/(n - 1)  #矩阵乘
        print('样本矩阵X的协方差C：\n', C)
        return C

    def _U(self):
        '''
        :return: 求X的降维转换矩阵U，shape=(n,k)，n是X的特征维度总数，k是降维矩阵的特征维度
        '''
        #先求协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.Cov)      #特征值赋值给a，对应特征向量赋值给b
        print('样本集的协方差矩阵C的特征值你：\n', a)
        print('样本集的协方差矩阵C的特征值对应的特征向量：\n', b)
        #给出特征值降序的Topk的索引序列
        ind_k= np.argsort(-1 * a)   #（-1*a从大到小）将a中的元素从小到大排列，提取其对应的index(索引)，然后输出到list。
        print('索引：', ind_k)
        #构件K阶降维转换矩阵U
        UT= [b[:,ind_k[i]] for i in range(self.k)]  #X[:,0]表示取二维数组中所有行的第0个元素。v[：，i]是对应于特征值w[i]的特征向量
        print('UT:', UT)
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U：\n'%self.k, U)
        return U

    def _Z(self):
        '''
        :return: 按照Z=XU求降为矩阵Z,shape=(n, k),n是样本总数，k是降维矩阵中特征维度总数.
        '''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维据矩阵Z：\n', Z)
        return Z

if __name__ == '__main__':
    '''10个样本3个特征的样本集， 行为样本， 列为特征数量'''
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
    k = X.shape[1] - 1
    print('样本集10行3列，10个样本，每个样本3个特征：\n', X, 'shape:', X.shape)
    pca = CPCA(X, k)
