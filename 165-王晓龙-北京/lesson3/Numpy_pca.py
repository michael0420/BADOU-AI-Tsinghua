import numpy as np

X = np.array([[-1, 2, 66, -1],
              [-2, 6, 58, -1],
              [-3, 8, 45, -2],
              [1, 9, 36, 1],
              [2, 10, 62, 1],
              [3, 5, 83, 2]])  # 导入数据，维度为4
# 数据的特征
features = X.shape[1]
# 去中心化
avg= X.mean(axis=0)
X = (X-avg)
# 协方差矩阵
cov = np.dot(X.T,X)/X.shape[0]
# 求协方差矩阵的特征值和特征向量
# np.linalg.eig() ?
eig_vals, eig_vectors = np.linalg.eig(cov)
# 降序排列 并获得index
idx = np.argsort(-eig_vals)
#这里取降维 2
# 获得将维矩阵
components = eig_vectors[:,idx[:2]]
#print(idx)
#输出降维后的数据
print(np.dot(X,components_))