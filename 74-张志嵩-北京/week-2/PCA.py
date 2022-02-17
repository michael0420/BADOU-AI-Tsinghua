import numpy as np
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris

class PCA():
    def __init__(self,X,Y,K):
        self.X = X
        self.Y = Y
        self.K = K
        self.centerX = []
        self.C = []
        self.U = []

        self.centerX = self._centralized()
        self.C = self._Cov()
        self.U = self._U()
        self.newX = np.dot(self.X, self.U)

    def _centralized(self):
        mean = np.array([i.mean() for i in self.X.T])
        centerX = self.X - mean
        return centerX

    def _Cov(self):
        num = np.shape(self.X)[0]
        C = np.dot(self.centerX.T,self.centerX)/(num-1)
        return C

    def _U(self):
        lamda,vector =  np.linalg.eig(self.C)
        ind = np.argsort(-1*lamda)
        U = np.array(vector[:,ind[:self.K]])
        return U
    def _plot(self):
        import matplotlib.pyplot as plt
        red_x, red_y = [], []
        blue_x, blue_y = [], []
        green_x, green_y = [], []
        for i in range(len(self.centerX)):
            if self.Y[i] ==0:
                red_x.append(self.newX[i][0])
                red_y.append(self.newX[i][1])
            elif self.Y[i] ==1:
                blue_x.append(self.newX[i][0])
                blue_y.append(self.newX[i][1])
            elif self.Y[i] ==2:
                green_x.append(self.newX[i][0])
                green_y.append(self.newX[i][1])
        plt.scatter(red_x, red_y, c='r', marker='x')
        plt.scatter(blue_x, blue_y, c='b', marker='D')
        plt.scatter(green_x, green_y, c='g', marker='.')
        plt.show()


if __name__ == "__main__":
    x,y = load_iris(return_X_y=True)
    pca = PCA(x,y,2)
    pca._plot()