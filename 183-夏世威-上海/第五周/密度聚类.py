from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
iris = iris.data
'''
密度聚类只要看着两个参数:
eps:选取类簇的最大距离,超高该值,就认为不属于该类簇
min_samples:成为类簇的最小样本数
'''
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(iris)
result = dbscan.labels_

# show
'''
这里我们已经看了结果有三类才这样写的,在实际应用中,分类结果往往是不确定的
一般根据实验和经验,再结合应用场景,调整 DBSCAN中的eps 和 min_samples 以达到我们的预期
'''
x0 = iris[result == 0]
x1 = iris[result == 1]
x2 = iris[result == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="r")
plt.scatter(x1[:, 0], x1[:, 1], c="g")
plt.scatter(x2[:, 0], x2[:, 1], c="b")
plt.show()
