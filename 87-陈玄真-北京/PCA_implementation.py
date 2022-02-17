
# 该代码实现PCA类的相关功能
# Author: chen x.z.
# Data: 2021/12/06

# 导入数值计算库
import numpy as np

class Cpca(object):
    '''将样本矩阵SamplMat的维度降为num'''
    def __init__(self, SamplMat, num):
        '''降维前的样本矩阵'''
        self.SamplMat = SamplMat
        '''降维后的维度'''
        self.num = num
        '''矩阵中心化结果初始化'''
        self.centrMat = []
        '''协方差矩阵初始化'''
        self.covarMat = []
        '''变换矩阵'''
        self.TransMat = []
        '''降维后的结果矩阵'''
        self.ResMat = []

        # 成员函数调用
        self.centrMat = self._centralized()
        self.covarMat = self._covar()
        self.TransMat = self._transform()
        self.ResMat = self._dimReduct()

    '''数据去均值函数'''
    def _centralized(self):
        '''打印样本矩阵'''
        # print('样本矩阵SamplMat:\n', self.SamplMat)
        mean = []
        for k in range(self.SamplMat.shape[1]):
            mean.append(np.sum(self.SamplMat[:,k])/self.SamplMat.shape[0])
        self.centrMat = self.SamplMat - mean
        '''打印去中心化结果'''
        print('中心化结果centralized:\n', self.centrMat)
        return self.centrMat

    '''计算去均值结果的协方差矩阵'''
    def _covar(self):
        temp = np.transpose(self.centrMat)
        self.covarMat = np.dot(temp,self.centrMat)/len(self.SamplMat)
        print('协方差矩阵covarMat:\n', self.covarMat)
        return self.covarMat

    '''计算变换矩阵'''
    def _transform(self):
        lamda,x = np.linalg.eig(self.covarMat)
        print('协方差矩阵covarMat的特征值:\n', lamda)
        print('协方差矩阵covarMat的特征向量:\n', x)
        # 给出特征值升序排列的序列
        index = np.argsort(lamda)
        # 从后面就得到最大的K个特征值对应的索引,再倒序得到K个最大特征值由大到小排列时相应的索引.
        topK = []
        tempMat = []
        for i in range(self.num):
            topK.append(index[-(i+1)])
        # 构造转换矩阵
        tempMat = [x[:, topK[i]] for i in range(self.num)]
        self.TransMat = np.transpose(tempMat)
        print('变换矩阵TransMat:\n', self.TransMat)
        return self.TransMat

    '''计算降维结果'''
    def _dimReduct(self):
        self.ResMat = np.dot(self.SamplMat,self.TransMat)
        print('降维矩阵ResMat:\n', self.ResMat)
        return self.ResMat

# 主函数
if __name__=='__main__':
    '''大小为10特征数为3的样本集'''
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

    # 打印原始样本矩阵
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)

    # 基于主成分分析降维
    pca = Cpca(X, 2)



