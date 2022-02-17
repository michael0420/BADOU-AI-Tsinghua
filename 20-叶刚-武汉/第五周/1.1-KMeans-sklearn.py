"""
@GiffordY
应用sklearn框架提供的KMeans接口（默认是KMeans++算法），对数据进行聚类
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
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
     [0.1956, 0.4280]]

X = np.array(X)
print('X.shape = ', X.shape)

# 应用sklearn框架提供的KMeans接口，对数据集进行聚类
# 1、创建KMeans类的实例对象kmeans，聚成3类（默认采用KMeans++算法）
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters)

# 2、调用fit_predict()方法，对数据集聚类并返回每个样本的类别
y_pred = kmeans.fit_predict(X)
centroids_pred = kmeans.cluster_centers_
print('y_pred = ', y_pred)
print('centroids_pred = ', centroids_pred)

# 3、绘制数据散点图
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for i in range(n_clusters):
    points = np.array([X[j] for j in range(len(X)) if y_pred[j] == i])
    # 参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], s=7, marker='o')
# 绘制聚类中心
plt.scatter(centroids_pred[:, 0], centroids_pred[:, 1], c='r', s=100, marker='*')
# 设置标题
plt.title("KMeans-Basketball Data")
# 设置x轴和y轴
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
# 设置图例，"A", "B", "C"表示不同的三种类别
plt.legend(["A", "B", "C"])
# 显示图形
plt.show()

