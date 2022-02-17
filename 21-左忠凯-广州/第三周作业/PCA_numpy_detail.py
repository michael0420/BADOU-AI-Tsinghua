import numpy as np

#PCA算法，计算X矩阵的K阶降维矩阵Z

class CPCA(object):

    def __init__(self, X, K):
        # X为训练样本矩阵
        # K为X矩阵的降维阶数
        self.X = X
        self.K = K
        self.centrX = []    # 中心化以后的矩阵
        self.COV = []       # 矩阵X的协方差矩阵
        self.U = []        # 特征向量矩阵
        self.Z = []       # 降维以后的矩阵

        self.centrX = self.central()
        self.COV = self.cov()
        self.U = self._U()
        self.Z = self._Z()


    '''样本矩阵中心化'''
    def central(self):
        centerX = []
        print("样本矩阵X:\n{}\n".format(self.X))
        mena = np.array(np.mean(self.X, axis=0)) # 计算每一列的均值，axis=1计算每行的均值，
        print("样本集的特征均值:\n{}\n".format(mena))
        centerX = self.X - mena
        print("样本矩阵X的中心化矩阵:\n{}".format(centerX))
        return centerX

    '''求协方差矩阵'''
    def cov(self):
        sample_num = np.shape(self.centrX)[0]  # np.shape第1个数表示样本数量，第2个数表示样本维度
        cov_mat = np.dot(self.centrX.T, self.centrX)/(sample_num - 1) # 计算协方差矩阵，self.centrX.T为centrX的转置矩阵
        print("样本矩阵X的协方差矩阵:{}\n".format(cov_mat))
        return cov_mat

    '''求X的降维转换矩阵'''
    def _U(self):
        # 得到协方差矩阵的特征值和特征向量
        a,b = np.linalg.eig(self.COV)  #特征值赋值给a, 特征向量赋值给b
        print("样本集协方差矩阵特征值为:\n{}".format(a))
        print("样本集协方差矩阵特征向量为:\n{}".format(b))

        # 对特征值进行排序，只要top K个
        ind = np.argsort(-1 * a) # 乘-1的目的是实现从大到小的排列(argsort默认从小到大排列)

        # 得到K阶降维转换矩阵
        UT = [b[:, ind[i]] for i in range(self.K)]  # 需要读取所有行对应的K列特征，这里采用矩阵切片方法读取，
                                                    # b[:,ind[i]]，','好前面是截取的起始行和终止行，这里':'前后为空，表示读取所有的行,
                                                    # 也就是所有的样本。后面的ind[i]表示依次读取前ind[i]列，一共读取K列
        U = np.transpose(UT)                        # 对UT进行转置
        print("K阶降维矩阵为:\n{}".format(U))
        return U

    '''得到样本矩阵X最终的降维矩阵，也就是用样本矩阵X乘以K阶降维矩阵，得到样本矩阵在K阶降维矩阵上的映射'''
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print("X Shape:{}".format(np.shape(self.X)))
        print("U Shape:{}".format(np.shape(self.U)))
        print("Z Shape:{}".format(np.shape(Z)))
        print("样本矩阵X降维后的矩阵Z:\n{}".format(Z))
        return Z

if __name__ == '__main__':
    '''10样本，3特征的样本集合X'''
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
    print("样本集为:\n{}".format(X))
    K = np.shape(X)[1] - 1  # 减少1维，也就是减低1维
    pca = CPCA(X, K)

