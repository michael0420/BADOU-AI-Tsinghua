#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于PCA降维后的Kmeans聚类
"""
from sklearn.cluster import KMeans
from sklearn.datasets._base import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def kmeans_sklearn(x,K):
    model = KMeans(K)
    y_pred = model.fit_predict(x)
    return y_pred

def show(reduced_x,y):
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()

def pca_sklearn(x):
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(x)  # 训练
    newX = pca.fit_transform(x)  # 降维后的数据
    return newX

if __name__ == '__main__':
    x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签
    newx = pca_sklearn(x)
    K = 3
    y_predit = kmeans_sklearn(newx, K)
    # plt.scatter(newx[:,0],newx[:,1],y_predit)
    # plt.show()
    show(newx,y_predit)
    show(newx, y)

    # plt.subplot(1,2,1)
    # plt.scatter(newx[:,0],newx[:,1],y_predit)
    # plt.subplot(1, 2, 2)
    # plt.scatter(newx[:, 0], newx[:, 1], y)
    # plt.show()
