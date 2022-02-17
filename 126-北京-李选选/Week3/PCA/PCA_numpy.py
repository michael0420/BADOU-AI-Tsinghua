import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        X = X - X.mean(axis=0)
        print('中心化后的X:\n',X)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        print('协方差矩阵：\n',self.covariance)
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        print('特征矩阵：\n',eig_vectors)
        # print(eig_vals)
        # print(eig_vectors)
        idx = np.argsort(-eig_vals)
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        print('转换矩阵：\n',self.components_)
        return np.dot(X, self.components_)


pca = PCA(n_components=3)
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
newX = pca.fit_transform(X)
print('结果：\n',newX)
