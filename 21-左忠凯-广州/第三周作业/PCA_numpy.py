import numpy as np

#PCA算法，计算X矩阵的K阶降维矩阵Z

class PCA():

    def __init__(self, n_components):
        self.n_components = n_components

    '''样本矩阵中心化'''
    def fit_transform(self, X):

        self.n_features = X.shape[1] # X样本矩阵的维数
        print("X样本维数:{}\r\n".format(self.n_features))

        # 1、样本中心化
        x_mean = X - np.array(np.mean(X, axis=0)) # 样本中心化
        print("中心化以后:\n{}".format(x_mean))

        # 2、获取协方差矩阵
        self.cov = np.dot(x_mean.T, x_mean) / X.shape[0]

        # 3、得到协方差矩阵对应的特征值和特征向量
        eig_vals,eig_vectors = np.linalg.eig(self.cov)
        print("特征值为:\n{}".format(eig_vals))
        print("特征向量为:\n{}".format(eig_vectors))

        # 5、得到降序排列的特征值，
        idx = np.argsort(-1 * eig_vals) # 乘-1的目的是实现从大到小的排列(argsort默认从小到大排列)
        print("特征值排序:\n{}".format(idx))

        # 6、K阶降维矩阵
        self.components = eig_vectors[:, idx[:self.n_components]]   # 技巧：读取矩阵的某些指定列[:,2]，表示角标为2的哪一列，
                                                                    # [:,[0,3]]表示读取角标为0和3的哪一列
        print("降维矩阵:{}\n".format(self.components))

        # 7、对X进行降维
        return np.dot(x_mean, self.components)

if __name__ == '__main__':
    '''5样本，4(维度)特征的样本集合X'''
    X = np.array([[-1, 2, 66, -1],
                  [-2, 6, 58, -1],
                  [-3, 8, 45, -2],
                  [1, 9, 36, 1],
                  [2, 10, 62, 1],
                  [3, 5, 83, 2]])
    print("样本集为:\n{}".format(X))
    pca = PCA(2)   # 降到2维
    newX = pca.fit_transform(X)
    print("降维后的X:\n{}".format(newX))

