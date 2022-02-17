# -*- coding: utf-8 -*-

import cv2 as cv
from sklearn.decomposition import PCA
import sklearn.decomposition as dp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._base import load_iris

class sklearn_PCA():
    # not understand
    def sklearn_PCA(self):
        X = np.array([[1,2,3,4], [2,4,6,8], [88,66,45,-1], [8,4,22,1], [3,9,66,25]])
        pca = PCA(n_components=2)
        # pca.fit(X) //information is that pca have no fit()
        X_new = pca.fit_transform(X)
        # print(pca.explained_variance_ratio_) //information is as same as fit()
        print(X_new)

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X)/X.shape[0]
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals)
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        return np.dot(X, self.components_)

class load_IRIS():
    def load_IRIS(self):
        x, y = load_iris(return_X_y=True)
        pca = dp.PCA(n_components=2)
        reduced_x = pca.fit_transform(x)
        red_x, red_y = [], []
        blue_x, blue_y = [], []
        green_x, green_y = [], []
        for i in range(len(reduced_x)):
            if y[i]==0:
                red_x.append(reduced_x[i][0])
                red_y.append(reduced_x[i][1])
            elif y[i]==1:
                blue_x.append(reduced_x[i][0])
                blue_y.append(reduced_x[i][1])
            else:
                green_x.append(reduced_x[i][0])
                green_y.append(reduced_x[i][1])
        plt.scatter(red_x, red_y, c='r',marker='X')
        plt.scatter(blue_x, blue_y, c='b', marker='D')
        plt.scatter(green_x, green_y, c='g', marker='.')
        plt.show()

class pca_Numpy_Details():
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        print('样本矩阵X：\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值：\n', mean)
        centrX = self.X - mean
        print('样本矩阵X的中心化：\n', centrX)
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值\n', a)
        print('样本集的协方差矩阵C的特征向量\n', b)
        ind = np.argsort(-1*a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U\n'%self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z\n', Z)
        return Z

class hist_Test():
    def hist_Gray(self):
        img = cv.imread("lenna.png", 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # plt.figure()
        # plt.hist(gray.ravel(),256)
        # plt.show()

        hist = cv.calcHist([gray], [0], None, [256], [0,256])
        plt.figure()
        plt.title("hist")
        plt.xlabel("hist classes")
        plt.ylabel("pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
    def hist_Color(self):
        img = cv.imread("lenna.png")
        cv.imshow("original image", img)
        cv.waitKey(0)

        chans = cv.split(img)
        colors = ("b", "g", "r")
        plt.figure()
        plt.title("hist color")
        plt.xlabel("Bins")
        plt.ylabel("Pixels")
        for(chan, color) in zip(chans, colors):
            hist = cv.calcHist([chan], [0], None, [256], [0,256])
            plt.plot(hist, color = color)
            plt.xlim([0, 256])
        plt.show()

class hist_Equalize():
    def hist_equalize(self):
        img = cv.imread("lenna.png", 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dst = cv.equalizeHist(gray)
        # plt.figure()
        # plt.hist(dst.ravel(), 256)
        # plt.show()

        # hist = cv.calcHist([dst], [0], None, [256], [0, 256])
        # plt.plot(hist)
        # plt.show()

        cv.imshow("hist equalize", np.hstack([gray, dst]))
        cv.waitKey(0)

    def color_equalize(self):
        img = cv.imread("lenna.png", 1)
        cv.imshow("original img", img)
        cv.waitKey(0)
        (b, g, r) = cv.split(img)
        bH = cv.equalizeHist(b)
        gH = cv.equalizeHist(g)
        rH = cv.equalizeHist(r)

        result = cv.merge((bH, gH, rH))
        cv.imshow("color_E", result)
        cv.waitKey(0)


if __name__ == '__main__':
     sklearn_PCA().sklearn_PCA()

    # pca = PCA(n_components=2)
    # X = np.array([[-1,2,66,1], [-2,6,58,-1], [-3,8,45,2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
    # X_new = pca.fit_transform(X)
    # print(X_new)

    # load_IRIS().load_IRIS()

    # X = np.array([[10,15,29],
    #              [15,46,13],
    #              [24,35,46],
    #              [1,2,3],
    #              [11,22,33],
    #              [2,3,4],
    #              [22,33,44],
    #              [4,5,6],
    #              [44,55,66],
    #              [12,23,34]])
    # K = np.shape(X)[1] - 1
    # print('样本集（10行3列，10个样例，每个样例3个特征）:\n', X)
    # pca = pca_Numpy_Details(X, K)

    # hist_Test().hist_Gray()
    # hist_Test().hist_Color()
    # hist_Equalize().hist_equalize()
    # hist_Equalize().color_equalize()