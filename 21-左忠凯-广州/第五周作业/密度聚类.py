import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

# 加载数据
iris = datasets.load_iris() # 加载鸢尾花数据
X = iris.data[:, :4] # 只取4个维度的数据
print(X.shape)

'''
DBSCAN(eps, min_samples)函数参数
    eps:领域半径，样本之间的距离不能超过邻域半径表示处于同一个簇
    min_samples：一个簇里面最少的样本数量
'''
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit_predict(X)
label_pred = dbscan.labels_  # 获取聚类后的数据标签，-1表示噪点
print(label_pred)

# 绘制图像
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

# scatter绘制点图
plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c='green', marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='+', label='label2')

plt.xlabel('speal length')
plt.ylabel('speal width')
plt.legend(loc=2) # 左上角添加提示
plt.show()






