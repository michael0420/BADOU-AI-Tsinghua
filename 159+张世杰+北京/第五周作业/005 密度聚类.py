# encoding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()  # 字典行书的数据
X = iris.data[:, :]  # 获得鸢尾花数据的data数据， 进行聚类：数据为150*4矩阵，150个数据，4个特征
'''调用接口进行聚类'''
dbscan = DBSCAN(eps=0.4, min_samples=9)  # 定义计算函数， 参数为：半径为0.4 ; 最小点数为9
dbscan.fit(X)  # 执行聚类

label_pred = dbscan.labels_
print(label_pred)
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

