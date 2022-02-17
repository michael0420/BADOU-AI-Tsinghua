from sklearn.cluster import KMeans

'''
数据集X，二维矩阵，第一列为球员每分钟助攻数，第二列为球员每分钟的分数
'''
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

'''
K-means聚类
KMeans(n_clusters, n_init, max_iter)，函数参数如下：
    返回值：K-Means聚类模型
    
    参数：
    n_clusters：需要聚出来的簇数，也就是K值，默认为8
    n_init：用不同质心初始化值运行的次数，最终返回最优结果，默认10
    max_iter：迭代次数，默认300
'''
clf = KMeans(n_clusters=3)  # 创建分类器对象
#clf_fit = clf.fit(X)
#y_pred = clf.predict(X)
y_pred = clf.fit_predict(X) # 加载数据集X，并将聚类结果也就是标签赋值给y_pred。fit_predict用训练器数据X拟合分类器模型，并对训练器
                            # 数据X进行预测，效果和分别调用fit和predict函数一样

print(clf)
print("cluster centers", clf.cluster_centers_) # 输出3个簇的中心
print("y_pred =", y_pred)

# 绘制图形
import numpy as np
import matplotlib.pyplot as plt

# 从原始数据中得到X,Y数据，X就是第一列，Y就是第二列
x = [n[0] for n in X]
y = [n[1] for n in X]
print(x)
print(y)

# 绘制散点图，原始数据第一列为X，第二列为Y,同一标签的数据用同一个种颜色
plt.scatter(x, y, c=y_pred, marker='x') # 绘制图形，颜色就用聚类结果，图像为x
plt.title("Kmeans-Basketbol Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend(["A","B","C"])
plt.show()