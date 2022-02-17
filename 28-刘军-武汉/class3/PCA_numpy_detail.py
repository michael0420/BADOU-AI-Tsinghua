import numpy as np
class CPCA():
    def __init__(self,X,k):
        self.X = X
        self.k = k
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()
    def _centralized(self):
        mean = [np.mean(attr) for attr in self.X.T]
        centrx = self.X-mean
        return centrx
    def _cov(self):
        ns = self.X.shape[0]
        C = np.dot(self.centrX.T,self.centrX)/(ns-1)
        return C
    def _U(self):
        a,b = np.linalg.eig(self.C)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.k)]
        U = np.transpose(UT)
        return U
    def _Z(self):
        Z = np.dot(self.X, self.U)
        return Z
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
pca = CPCA(X, K)

