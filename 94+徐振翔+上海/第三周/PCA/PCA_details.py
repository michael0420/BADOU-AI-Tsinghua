# 实现PCA核函数,并提供阈值接口：1.使用自制函数 2.自制输入接口，可选k值或损失率阈值，默认为输入参数抵一维

import numpy as np
import os

class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''
    def __init__(self):
        '''
        Y = X @ K
        X:shape = (m,n):原始输入样本，m行，n个特征
        k:降维后的维数
        X_c:中心化后的X矩阵
        Cov:协方差矩阵
        K:特征向量组成的降维矩阵，
        Y:转换后的矩阵
        '''
        self.X = []
        self.k = 0
        self.X_c = []
        self.Cov = []
        self.K = []
        self.state = 0
        self.nk_compare = 1.1
        self.nk = 0


    def output(self):
        if self.state == 0:
            print("Error")
        else:
            print('-'*20)
            print("Input Matrix:")
            print(self.X)
            print("Output Matrix:")
            print(self.Y)
            print("original dim:{},target dim:{}".format(self.X.shape[1], self.k))
            if self.nk_compare < 1:
                print("nk:{},nk_compare:{}".format(self.nk, self.nk_compare))
            else:
                print("nk:{}".format(self.nk))

    def load(self, x, k=-1, nk_compare=1.1):
        '''
        X:shape = (m,n):原始输入样本，m行，n个特征
        k:降维后的维数,可选，默认为x的维数-1
        nk_compare:保留的信息阈值，默认不启用，输入小于1的值且未输入k时，会根据该阈值自动调整维数
        '''
        self.X = x
        if (not nk_compare == 1.1) and (k == -1):
            self.nk_compare = nk_compare
            self.k = -1
        elif k == -1:
            self.k = self.X.shape[1] - 1
        else:
            self.k = k

    def center(self):
        '''矩阵X中心化'''
        self.X_c = []
        # np求均值的方法 mean
        # print([np.mean(attr) for attr in self.X])
        # print('-'*20)
        # print([np.mean(attr) for attr in self.X.T])
        # print('-' * 20)
        # print(np.mean(self.X, axis=0))
        # print('-' * 20)
        # print(np.mean(self.X, axis=1))
        mean = np.mean(self.X, axis=0) #求样本中特征的平均值
        self.X_c = self.X - mean
        print("样本中心化:\n", self.X_c)

    def cov(self):
        # numpy.cov(m, y=None, rowvar=True)
        # m：array_like 包含多个变量和观测值的1-D或2-D数组。 m的每一行代表一个变量，每一列都是对所有这些变量的单一观察。
        # y：array_like，可选 另外一组变量和观察。 y具有与m相同的形式。
        # rowvar：布尔值，可选 如果rowvar为True（默认值），则每行代表一个变量X，另一个行为变量Y。否则，转换关系：每列代表一个变量X，另一个列为变量Y。
        self.Cov = np.cov(self.X_c, rowvar=False)
        print("协方差矩阵：\n", self.Cov)
        # 另一种算法：cov = XT @ X /(n-1)
        # # 样本集的样例总数
        # ns = np.shape(self.X_c)[0]
        # # 样本矩阵的协方差矩阵C
        # C = np.dot(self.X_c.T, self.X_c) / (ns - 1)
        # print("协方差矩阵：\n", C)

    def eig(self):
        '''
        Y = X @ K
        X:shape = (m,n):原始输入样本，m行，n个特征
        k:降维后的维数
        X_c:中心化后的X矩阵
        Cov:协方差矩阵
        K:特征向量组成的降维矩阵，
        Y:转换后的矩阵
        '''
        eigValMat = []
        eigVal = []
        eigVal, eigVector = np.linalg.eig(self.Cov)
        print("特征值\n", eigVal)
        print("特征向量\n", eigVector)
        # numpy.argsort(a, axis=-1, kind=’quicksort’, order = None)
        # 功能: 将矩阵a按照axis排序，并返回排序后的下标
        # 参数: a:输入矩阵， axis: 需要排序的维度
        # 返回值: 输出排序后的下标
        # 排序结果为从小到大
        eigValInd = np.argsort(- eigVal) #取负数排序实现由大到小排序
        print("特征值降序序列\n", eigValInd)
        if self.nk_compare < 1:
            k = 0
            sum = 0
            threshold = self.nk_compare * np.sum(eigVal)
            while (sum + eigVal[eigValInd[k]]) < threshold:
                sum += eigVal[eigValInd[k]]
                k += 1
            self.k = k + 1
        eigValMat = eigVal[eigValInd[:self.k]]
        eigValInd = eigValInd[:self.k]  # 截断不需要的特征值对应ID
        # eigVectorMat = eigVector[eigValInd[:self.k]]
        eigVectorMat = eigVector[:, eigValInd]  # 获取对应的特征向量矩阵
        print("保留的特征值矩阵\n", eigValMat)
        print("保留的特征向量矩阵(转换矩阵)\n", eigVectorMat)
        self.K = eigVectorMat
        self.nk = np.sum(eigValMat)/np.sum(eigVal)



    def getY(self):
        '''Y = X @ K'''
        self.Y = np.dot(self.X, self.K)
        print("降维后的矩阵({}维)\n{}".format(self.k, self.Y))


    def cal(self):
        '''计算'''
        if len(self.X) < 1:
            # 简单的过滤
            print("error")
            os.exit()
        self.state = 0
        self.center()
        self.cov()
        self.eig()
        self.getY()
        self.state = 1
        self.output()


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
    pca = CPCA()
    # pca.load(X)
    # pca.load(X, k=1)
    pca.load(X, nk_compare=0.9)
    # pca.load(X, nk_compare=0.7)
    pca.cal()






