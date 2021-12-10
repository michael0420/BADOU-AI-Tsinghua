import numpy as np


def PCA(data, k):
    # PCA: Select number of k principle components' eigenvectors with the largest k eigenvalues.

    # Subtract the sample mean from data.
    data = data - np.mean(data, axis=0)

    # Compute the covariance matrix of the data.
    sigma = np.cov(data, rowvar=0, bias=1)

    # Compute the eigenvalues and eigenvectors of the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)

    # Descend the eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the first k principle components.
    W = eigenvectors[:, :k]
    # PCA data after dimension reduction
    Z = np.dot(data, W)
    # Data return from PCA dimension to the start dimension
    Y = Z.dot(W.T)
    return Z, Y


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    # X = np.array([[10, 15, 29],
    #               [15, 46, 13],
    #               [23, 21, 30],
    #               [11, 9, 35],
    #               [42, 45, 11],
    #               [9, 48, 5],
    #               [11, 21, 14],
    #               [8, 5, 15],
    #               [11, 12, 21],
    #               [21, 20, 25]])
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
    # K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    z, y = PCA(X, 2)
    print(z)
