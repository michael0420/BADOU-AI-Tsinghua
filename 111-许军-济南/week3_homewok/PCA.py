# -- coding:utf-8 --
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
import numpy as np
from sklearn.datasets._base import load_iris

class PCA:
    def __init__(self,x,k):
        self.x = x # 原矩阵
        self.k = k #维度
        self.center_x = self._centralized() # 中心化矩阵
        self.cov_mat = self._convariance() # 协方差矩阵
        self.eigenvalues, self.fea_vector = self._eigenvalues() # 特征值和特征向量
        self.DM = self._DM() # 变换矩阵
        self.z = self._Dim()

    # 中心化
    def _centralized(self):
        mean = [np.mean(i) for i in self.x.T]
        center_x = x - mean
        return center_x
    # 协方差矩阵
    def _convariance(self):
        cov_mat = np.zeros((np.shape(self.center_x)[1],np.shape(self.center_x)[1]))
        for i in range(cov_mat.shape[0]):
            for j in range(cov_mat.shape[1]):
                cov_mat[i][j] = sum(np.multiply(self.center_x[:,i],self.center_x[:,j]))/(self.center_x.shape[0])
        return cov_mat

    # 求协方差矩阵的特征值，特征向量
    def _eigenvalues(self):
        eigenvalues,fea_vector = np.linalg.eig(self.cov_mat)
        return eigenvalues,fea_vector
    # 转换矩阵
    def _DM(self):
        eig_sort = np.argsort(-1*self.eigenvalues,axis=0)
        DM = [self.cov_mat[:,eig_sort[i]] for i in range(self.k)]
        return DM
    # 降维后的样本
    def _Dim(self):
        DM_  = np.transpose(self.DM)
        z = np.dot(self.x,DM_)
        return z

if __name__ == '__main__':
    x, y=load_iris(return_X_y=True)
    pca = PCA(x,3)
    print(pca.z)



