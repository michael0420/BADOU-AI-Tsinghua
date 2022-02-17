''''
用Numpy自实现CPA过程
'''

import numpy as np

class PCA_Diy():
    def __init__(self,n_components):
        self.n_components=n_components

    def transformerr(self,X):
        self.X=X
        self.centerX=[]
        self.Covariance=[]
        self.Vectors=[]
        self.NewResult=[]

        self._centralized()
        self._covariance()
        self._eigCalc()
        self.fit()

    def _centralized(self):
        # mean=[]
        # for attr in self.X.T:
        #     mean_attr=np.mean(attr)
        #     mean.append(mean_attr)
        mean=np.array([np.mean(attr) for attr in self.X.T])
        self.centerX=self.X-mean

    def _covariance(self):
        size=self.centerX.shape[0]
        self.Covariance=np.dot(self.centerX.T,self.centerX)/(size-1)
        print('协方差矩阵：\n',self.Covariance)

    def _eigCalc(self):
        a,b=np.linalg.eig(self.Covariance)
        ind=np.argsort(a*-1)
        _Vectors=[b[:,ind[i]] for i in range(self.n_components)]
        self.Vectors=np.transpose(_Vectors)
        #直接取列值组成转置矩阵
        # self.Vectors=b[:,ind[:self.n_components]]
        print("转换矩阵：\n",self.Vectors)

    def fit(self):
        self.NewResult=np.dot(self.X,self.Vectors)
        print("降维后结果：\n",self.NewResult)

# X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
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
pca=PCA_Diy(2)
pca.transformerr(X)




