import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取数据
data = np.genfromtxt('kmean_test_data.txt', delimiter=',')
# 聚类数量
k = 4
# 训练模型
model = KMeans(n_clusters=k)
model.fit(data)
# 分类中心点坐标
centers = model.cluster_centers_
# 预测结果
result = model.predict(data)
# 用不同的颜色绘制数据点
mark = ['or', 'og', 'ob', 'ok']
for i, d in enumerate(data):
    plt.plot(d[0], d[1], mark[result[i]])
# 画出各个分类的中心点
mark = ['*r', '*g', '*b', '*k']
for i, center in enumerate(centers):
    plt.plot(center[0], center[1], mark[i], markersize=20)

# 绘制簇的作用域
# 获取数据值所在的范围
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
cs = plt.contourf(xx, yy, z)
plt.show()

