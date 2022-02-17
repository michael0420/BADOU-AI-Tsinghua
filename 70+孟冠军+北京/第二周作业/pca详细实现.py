import numpy as np

#把4维降维为三维的
X = np.array([[10, 15, 29,22],[15, 46, 13,34],[23, 21, 30,23],[11, 9, 35,25],[42, 45, 11,34],
             [9, 48, 5,25],[11, 21, 14,28],[8, 5, 15,19],[11, 12, 21,17],[21, 20, 25,23]])
K = np.shape(X)[1] - 1   #k=3

#一：pca的调接口实现
from sklearn.decomposition import PCA
pca = PCA(n_components=K)
pca.fit(X)
X_new=pca.fit_transform(X)
print('pca调接口实现的降维矩阵X_new:\n',X_new)  

#二：pca的详细实现
#（1）将样本矩阵中心化
centrX = []
mean = np.array([np.mean(attr) for attr in X.T])  # 求特征均值
print('样本集的特征均值:\n', mean)
centrX = X - mean  ##得到中心化矩阵
print('样本矩阵X的中心化centrX:\n', centrX)

#（2）求中心化之后的协方差矩阵C
ns = np.shape(centrX)[0]#样本数量
C = np.dot(centrX.T, centrX) / (ns - 1) #样本矩阵的协方差矩阵C
print('样本矩阵X的协方差矩阵C:\n', C)

#（3）求X的降维转换矩阵U
a, b = np.linalg.eig(C)# 先求X的协方差矩阵C的特征值和特征向量
print('样本集的协方差矩阵C的特征值:\n', a)
print('样本集的协方差矩阵C的特征向量:\n', b)
ind = np.argsort(-1 * a)# 给出特征值降序的topK的索引序列
UT = [b[:, ind[i]] for i in range(K)]
U = np.transpose(UT)
print('%d阶降维转换矩阵U:\n' % K, U)

#（4）得到降维矩阵Z
Z = np.dot(X, U)
print('X shape:', np.shape(X))
print('U shape:', np.shape(U))
print('Z shape:', np.shape(Z))
print('样本矩阵X的降维矩阵Z:\n', Z)


