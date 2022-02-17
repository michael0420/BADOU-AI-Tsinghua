# 1.手动详细实现numpy

import numpy as np
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
K = np.shape(X)[1] - 1  # np.shape(X)是一个由样本数和维度组成的数组，[1]即是第二个元素---维度
print(K)
print('样本集(10行3列，10个样例，每个样例3个特征):\n',X)

# 进行样本中心化
mean=np.array([np.mean(attr) for attr in X.T])  # X.T 获得X 的转置矩阵，然后求均值
print('样本集的特征均值:\n',mean)
centrX= X-mean   # 每行对应减mean

#  协方差矩阵
ns=np.shape(X)[0]  # 样本总数
C = np.dot(centrX.T, centrX)/ns  # 中心化之后的协方差矩阵公式
print('样本矩阵X的协方差矩阵C:\n', C)
a,b = np.linalg.eig(C) # 把特征值赋给a,把特征向量赋给b
print('样本集的协方差矩阵C的特征值:\n', a)
print('样本集的协方差矩阵C的特征向量:\n', b)
ind = np.argsort(-1*a)
print(ind)
UT = [b[:,ind[i]] for i in range(K)]
U = np.transpose(UT)  # 新的向量空间的基
print('%d阶降维转换矩阵U:\n'%K, U)

print(X)
print(U)

Z = np.dot(centrX,U)            # 进行降维 Z=P*X
print('X shape:', np.shape(X))
print('U shape:', np.shape(U))
print('Z shape:', np.shape(Z))
print('(numpy手动实现)样本矩阵X的降维转置矩阵Z:\n', Z)
print("----------------------------分割线----------------------------\n\n")

# 2.sklearn 实现
from sklearn.decomposition import PCA
T = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9,  35],
              [42, 45, 11],5
              [9,  48, 5],
              [11, 21, 14],
              [8,  5,  15],
              [11, 12, 21],
              [21, 20, 25]])  # 导入数据，维度为3
pca = PCA(n_components=2)   # 降到2维
pca.fit(T)                  # 训练
newX=pca.fit_transform(T)   # 降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  # 输出贡献率
print("(sklearn)样本矩阵T的降维转置矩阵Z:\n",newX)                  # 输出降维后的数据
print("----------------------------分割线----------------------------\n\n")

# 3.numpy函数调用
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        print("协方差矩阵：\n",self.covariance)
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        print(eig_vals,"\n", eig_vectors)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        print(idx)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        print(self.components_)
        print(X)
        # 对X进行降维

        return np.dot(X, self.components_)


# 调用
pca = PCA(n_components=2)
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
newX = pca.fit_transform(X)
print("(numpy函数调用)样本矩阵T的降维转置矩阵Z:\n",newX)  # 输出降维后的数据