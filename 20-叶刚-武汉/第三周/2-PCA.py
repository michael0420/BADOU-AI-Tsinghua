"""
@author: GiffordY
Principal Component Analysis(PCA) algorithm
"""
import numpy as np
import sklearn.decomposition as dp
from sklearn import datasets


def pca_algo(X, n_components: int):
    """
    Principal Component Analysis(PCA) algorithm
    :param X: array_like, original matrix, row: samples; col: features
    :param n_components:
    :return:
    """
    # 1.Zero averaging (centralization) of the original data X
    X = X - X.mean(axis=0)
    # 2.Calculate covariance matrix
    Cov = np.dot(X.T, X) / X.shape[0]
    # 3.Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(Cov)
    # 4.Sort the eigenvalues from large to small, and return the corresponding indices
    indices = np.argsort(-1 * eigenvalues)
    # 5.Select the eigenvectors corresponding to the largest K eigenvalues to form the transformation matrix
    W = eigenvectors[:, indices[0: n_components]]
    # 6.Use the transformation matrix to reduce the dimensionality of the input matrix
    Y = np.dot(X, W)
    # 7.Evaluation indicator: The amount of information retained by the top k eigenvalues
    ratio = np.sum(eigenvalues[indices[0: n_components]]) / np.sum(eigenvalues)
    return Y, ratio


def pca_from_sklearn(X, n_components: int):
    pca = dp.PCA(n_components)
    pca.fit(X)
    Y = pca.fit_transform(X)
    per_ratios = pca.explained_variance_ratio_
    return Y, per_ratios


if __name__ == '__main__':
    src_datas, labels = datasets.load_iris(return_X_y=True)
    dst_datas, ratio = pca_algo(src_datas, 2)
    print('dst_datas = ', dst_datas)
    print('ratio = ', ratio)

    dst_datas_2, ratio_2 = pca_from_sklearn(src_datas, 2)
    # print('dst_datas_2 = ', dst_datas_2)
    print('ratio_2 = ', ratio_2)

