# encoding: utf-8
import numpy as np
from sklearn.decomposition import PCA


def pca_ex(arr):
    pca = PCA(n_components=2)
    pca.fit(arr) # 这行代码可以不要
    newX = pca.fit_transform(arr) # 降维处理，此处默认对列维度进行降维
    print(pca.explained_variance_ratio_)  # 贡献率：特征值的占特征值总和的百分比，也就是转换后的数据集的选出来的两维方差之和占总方差的百分比
    print(newX)
    return newX


X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
pca_ex(X)
