#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

class k_MEANS():
    def k1(self):
        img = cv.imread("lenna.png", 0)
        rows, cols = img.shape[:]
        data = img.reshape(rows*cols, 1)

        # providing array datasets
        data = np.float32(data)

        # Asure the stop condition
        stop_IF = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Asure methods that sets initial data center
        center_TEMP = cv.KMEANS_RANDOM_CENTERS

        # call cv2.kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
        '''
        @brief Implementing a algorithm that finds the centers of dataArray and groups it
        
        @param data: Inout dataArray
        @param K: The wanted class number of dataArray
        @param bestLabels: Input/output integer array that stores the cluster indices for every sample//???,classified index
        @param criteria: A algorithm of setting stop conditions(1.iterations 2.epsilon)
        @param attempts: The number of executing algorithm
        @param flags: The methods of selecting initial datacenters
        
        @return compactness, labels, centers
        The best (minimum) value is chosen and the corresponding labels and the compactness value are returned by the function
        
        '''
        a, b, c = cv.kmeans(data, 4, None, stop_IF, 10, center_TEMP)
        dst = b.reshape(img.shape[0], img.shape[1])

        plt.rcParams['font.sans-serif'] = ['SimHei']

        titles = ("原始图", "聚类图")
        images = [img, dst]

        plt.imshow(images[0], 'gray')
        plt.title(titles[0])
        plt.show()

        plt.imshow(images[1], 'gray')
        plt.title(titles[1])
        plt.show()

    def k2(self):
        """
        @brief Test KMeans function through basketball dataset

        @return: a array consists of index of the grouped dataset's points belong to
        """

        X = [[0.0888, 0.5885],
             [0.1399, 0.8291],
             [0.0747, 0.4974],
             [0.0983, 0.5772],
             [0.1276, 0.5703],
             [0.1671, 0.5835],
             [0.1306, 0.5276],
             [0.1061, 0.5523],
             [0.2446, 0.4007],
             [0.1670, 0.4770],
             [0.2485, 0.4313],
             [0.1227, 0.4909],
             [0.1240, 0.5668],
             [0.1461, 0.5113],
             [0.2315, 0.3788],
             [0.0494, 0.5590],
             [0.1107, 0.4799],
             [0.1121, 0.5735],
             [0.1007, 0.6318],
             [0.2567, 0.4326],
             [0.1956, 0.4280]
             ]

        # set the first parameter that number of grouped clusters
        clf = KMeans(n_clusters=3)
        print(clf)

        # return the result of grouped clusters
        y_pred = clf.fit_predict(X)
        print("y_pred = ", y_pred)

        # paint image
        x = [n[0] for n in X]
        print(x)
        y = [n[1] for n in X]
        print(y)

        plt.scatter(x, y, c=y_pred, marker='x')
        plt.title("Kmeans-Basketball Data")
        plt.xlabel("assists_per_minute")
        plt.ylabel("points_per_minute")
        plt.legend(["A", "B", "C"])
        plt.show()






if __name__=='__main__':
    # k_MEANS().k1()
    k_MEANS().k2()

