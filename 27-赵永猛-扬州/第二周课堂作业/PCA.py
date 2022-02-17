import numpy as np


class CPCA():

    def __init__(self, array, K):
        self.array = array
        self.K = K
        self.C = self.Covariance_matrix()
        self.V = self.dec_cov_mat()
        self.U = self.topk_fea_mat()
        self.Z = self._Z()

    def Covariance_matrix(self):
        print('样本矩阵X:\n', self.array)
        col_mean = np.mean(self.array, axis=0) # mean value of each columns
        print('样本矩阵的均值：\n', col_mean)
        C = self.array - col_mean.T # decentralized matrix
        print('样本矩阵的去中心化\n', C)
        return C

    def dec_cov_mat(self):
        num_sample = len(self.array.T[0])
        print('样本数：\n', num_sample)
        V = np.dot(self.C.T, self.C)/(num_sample-1)
        print('去中心化的协方差矩阵：\n', V)
        return V

    def topk_fea_mat(self):
        eigenvalue, feature_vector = np.linalg.eig(self.V)
        print('协方差矩阵特征值及对应的特征向量：\n', eigenvalue, feature_vector)
        ind = np.argsort(-1 * eigenvalue)
        print('索引', ind)
        # 构建K阶降维的降维转换矩阵U
        UT = [feature_vector[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('前k个特征值对应的特征向量矩阵：', U)
        return U

    def _Z(self):
        Z = np.dot(self.array, self.U)
        print('array shape:', np.shape(self.array))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('\n', Z)
        return Z


if __name__=='__main__':
    '''10样本3特征的样本集, 行为样例，列为特征维度'''
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
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)




