import numpy as np
class pca():
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centreX = []
        self.C = []
        self.U = []
        self.Z = []
        self.centreX = self._centralized()
        self. C = self._cov()
        self.U = self._U()
        self.Z = self._Z()
    def _centralized(self):
        print('样本矩阵', self.X)
        cnetrX = []
        mean = np.array([np.mean(a) for a in self.X.T])
        print('样本每列均值', mean)
        cnetrX = self.X - mean
        print('样本中心化', cnetrX)
        return cnetrX
    def _cov(self):
        ns = np.shape(self.centreX)[0]
        C = np.dot(self.centreX.T, self.centreX)/(ns-1)
        return C
    def _U(self):
        a, b = np.linalg.eig(self.C)
        print(f'特征值{a}')
        print(f'特征向量{b}')
        ind = np.argsort(-a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        print(UT)
        U = np.transpose(UT)
        print(f'{self.K}阶降维矩阵U', U)
        return U
    def _Z(self):
        Z =np.dot(self.X, self.U)
        print(Z)
        return Z










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
K = np.shape(X)[1]-1
print('降维前的样本', X)
PCA = pca(X, K)
