"""
应用sklearn框架提供的DBSCAN（密度聚类）算法接口，对鸢尾花数据集进行聚类
"""

from sklearn import datasets
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :4]  # 表示取所有样本的前4个特征
print('X.shape = ', X.shape)

# 使用前2个特征绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

# 对鸢尾花数据集进行密度聚类
"""
密度聚类的原理：
(1)它由一个任意未被访问的点开始，然后探索这个点的 epsilon邻域，如果epsilon邻域里有足够的点(min_samples)，
则建立一个新的聚类，否则这个点被标签为杂音;
(2)注意，这个杂音点之后可能被发现在其它点的 epsilon 邻域里，而该epsilon邻域可能有足够的点，届时这个点会被加入该聚类中。
"""
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

# 绘制结果
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
