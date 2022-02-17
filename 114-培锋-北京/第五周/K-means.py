'''
k-means聚类算法(基于欧氏距离)https://blog.csdn.net/PyRookie/article/details/81915078
分析流程：
第一步，确定K值，即将数据集聚集成K个类簇或小组。
第二步，从数据集中随机选择K个数据点作为质心（Centroid）或数据中心。
第三步，分别计算每个点到每个质心之间的距离，并将每个点划分到离最近质心的小组。
第四步，当每个质心都聚集了一些点后，重新定义算法选出新的质心。（对于每个簇，计
算其均值，即得到新的k个质心点）
第五步， 迭代执行第三步到第四步，直到迭代终止条件满足为止（聚类结果不再变化）
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#生成样本数据
X,Y = make_blobs(n_samples=100,#100个样本
                 n_features = 2,#每个样本2个属性
                 centers = 4,#4个中心点
                 random_state = 1)

color = ['red','pink','orange','gray']
fig,axi = plt.subplots(1)
for i in range(4):
    axi.scatter(X[Y==i,0],X[Y==i,1],marker = 'o',s=8 ,c = color[i])

plt.title('produciton data')
plt.show()


clf = KMeans(n_clusters = 4,random_state=0)
cluster = clf.fit_predict(X)

print(clf)
print(cluster)


# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=cluster, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
#plt.xlabel("assists_per_minute")
#plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()