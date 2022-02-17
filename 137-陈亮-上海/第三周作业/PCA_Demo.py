import cv2
import numpy as np

class PCA(object):
    def __init__(self,X,K):
        # 定义空矩阵
        self.X = X
        self.K = K
        self.centerX = []
        self.Cov = []
        self.new_feture = []
        self.output = []

        # 对每个矩阵赋值
        self.centerX = self._center()
        self.Cov = self._Cov()
        self.new_feture = self.new_feture_U()
        self.output = self._output()


    def _center(self):
        '''求中心矩阵'''
        mean = np.array([np.mean(item) for item in self.X.T])
        center = self.X - mean
        return center

    def _Cov(self):
        '''卷积公式：C = np.dot(Z.T,Z)/(m-1)'''
        sum_X = np.shape(self.centerX)[0]
        Cov = np.dot(self.centerX.T,self.centerX)/(sum_X-1)
        return Cov

    def new_feture_U(self):
        '''将特征值和特征向量取出，按从大到小排序，组成所需新的特征向量空间'''
        a,b = np.linalg.eig(self.Cov)
        seq = np.argsort(-1*a)
        U = [b[:,seq[i]] for i in range(self.K)]
        new_feture_U = np.transpose(U)
        return new_feture_U

    def _output(self):
        '''将样本空间映射到特征向量空间内'''
        _output = np.dot(self.X,self.new_feture)
        print(_output)
        return _output

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
