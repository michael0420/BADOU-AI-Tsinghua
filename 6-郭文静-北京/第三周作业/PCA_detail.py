#coding=utf-8
"""
PCA对鸢尾花数据进行降维
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris



def pca_detail(X,K):
    # 1: 零均值，去中心化 x-mean(x)
    meanV=np.array([np.mean(i) for i in X.T])
    centerX=X-meanV
    # 2: 求协方差矩阵 cxT*cx
    n=np.shape(centerX)[0]
    covvar=np.dot(centerX.T,centerX)/(n-1)
    # 3. 求特征值和特征向量
    eigen,eigen_vector=np.linalg.eig(covvar)
    #4. 降维为K列
    index=np.argsort(-1*eigen)
    UT=[eigen_vector[:,index[i]] for i in range(K)]
    U=np.transpose(UT)
    #5. 求降维矩阵Z=X*U
    Z=np.dot(X,U)
    return Z

(x,y)=load_iris(return_X_y=True)

# print(x.shape)
# print(y.shape)
K = 2
x1=pca_detail(x[0:50,:],K)
x2=pca_detail(x[50:100,:],K)
x3=pca_detail(x[100:150,:],K)

plt.scatter(x1[:,0],x1[:,1],c='r',marker='x')
plt.scatter(x2[:,0],x2[:,1],c='b',marker='D')
plt.scatter(x3[:,0],x3[:,1],c='g',marker='.')
plt.show()



