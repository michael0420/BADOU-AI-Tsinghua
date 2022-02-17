#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2021/12/13 20:39:57
@Author : LuZhanglin

PCA详细实现

'''


import numpy as np


class NPCA:
    """
    Step 1: 数据中心化
    Step 2: 协方差矩阵
    Step 3: 求特征值和特征向量
    Step 4: 降维

    Parameters
    ----------
    X : np.ndarray
        需降维的数据, shape (n_samples, n_feature)
    K : int
        目标维数
    """
    def __init__(self, X, K) -> None:
        self.X = X
        self.K = K
        """样本特征中心化"""
        self.X_center = self.X - self.X.mean(axis=0)
        """求协方差矩阵"""
        self.cov_mat = np.dot(self.X_center.T, self.X_center) / (self.X.shape[0] - 1)
        """求K个主成分特征向量矩阵"""
        eigvals, elgvectors = np.linalg.eig(self.cov_mat)
        print("特征值：", eigvals)

        # 按特征值高低排序
        ind = np.argsort(-eigvals)

        # 取前K个特征向量
        self.eigv_K = elgvectors[:, ind[:self.K]]
        """求降维矩阵"""
        self.Z = np.dot(self.X_center, self.eigv_K)


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    # test1
    X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
    pca = PCA(1)
    reduced_x = pca.fit_transform(X)
    # 二维降到1维
    pca = NPCA(X, 1)
    Xnew = pca.Z
    print("Xnew by detail code:\n", Xnew, '\nXnew by sklearn PCA:\n', reduced_x)
    print("test1-两种方法实现效果一样：", (np.abs(Xnew - reduced_x)<1e-6).all())
    # test2
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

    pca = PCA(1)
    reduced_x = pca.fit_transform(X.copy())
    reduced_x_restore = pca.inverse_transform(reduced_x)

    pca = NPCA(X, 1)
    Xnew = pca.Z
    # 发现两个结果正负号不一样
    print("Xnew by detail code:\n", Xnew,  '\nXnew by sklearn PCA:\n', reduced_x)
    print("test2-两种方法的结果符号相反：", (np.abs(Xnew + reduced_x) < 1e-6).all())

    Xnew_restore = Xnew.dot(pca.eigv_K.T) + X.mean(0)
    print("test2-两种方法的结果进行维度还原后相等：{}，说明两种方法实现效果还是一样的".format((np.abs(reduced_x_restore - Xnew_restore) < 1e-6).all()))