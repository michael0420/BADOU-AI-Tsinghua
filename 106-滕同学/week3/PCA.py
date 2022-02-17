# author: orea

"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""
 
import numpy as np
 

class CPCA(object):

    def __init__(self, X, K):
        '''
        :param X,训练样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []
        
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()
        
    def _centralized(self):
        '''矩阵X的中心化'''
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX
        
    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        return C
        
    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        a,b = np.linalg.eig(self.C)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U
        
    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        return Z
        
        
if __name__=='__main__':
  import matplotlib.pyplot as plt
  from sklearn import datasets

  x, y = datasets.load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签
  my_pca = CPCA(x, K=2) #对原始数据进行降维，保存在reduced_x中
  reduced_x = my_pca.X
  red_x, red_y = [],[]
  blue_x, blue_y = [],[]
  green_x, green_y = [],[]
  for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

  plt.scatter(red_x, red_y, c='r', marker='x')
  plt.scatter(blue_x, blue_y, c='b', marker='D')
  plt.scatter(green_x, green_y, c='g', marker='.')
  plt.show()
