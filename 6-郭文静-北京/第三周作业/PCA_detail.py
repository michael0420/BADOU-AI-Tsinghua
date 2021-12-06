#coding=utf-8
"""
PCA对鸢尾花数据进行降维
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import sklearn.decomposition as dp


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
    print(U)
    #5. 求降维矩阵Z=X*U
    Z=np.dot(X,U)
    return Z

(x,y)=load_iris(return_X_y=True)

# print(x.shape)
# print(y.shape)
K = 2
reduced_x=pca_detail(x,K)

pca=dp.PCA(n_components=2)
pca.fit(x)
print(pca.components_)
print(pca.explained_variance_)
print(pca.singular_values_)
#reduced_x=pca.fit(x)
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()

x1=pca_detail(x[0:50,:],K)
x2=pca_detail(x[50:100,:],K)
x3=pca_detail(x[100:150,:],K)

plt.scatter(x1[:,0],x1[:,1],c='r',marker='x')
plt.scatter(x2[:,0],x2[:,1],c='b',marker='D')
plt.scatter(x3[:,0],x3[:,1],c='g',marker='.')
plt.show()



