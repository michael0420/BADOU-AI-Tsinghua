#coding:utf-8

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''调用sklearn中的KMeans
提供运动员数据
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
     [0.1956, 0.4280]
    ]
print('X:\n', X)

#接口调用
clusters = KMeans(n_clusters=3)     #k=3
y_pred = clusters.fit_predict(X)    # 喂数据

print('clusters:\n', clusters)     #输出完整Kmeans函数，包括很多省略参数
print('y_pred:\n', y_pred)         #输出聚类预测结果

#绘制散点图
x = [n[0] for n in X]
y = [n[1] for n in X]
print('x:\n', x)
print('y:\n', y)

plt.scatter(x, y, c=y_pred, marker='x')

plt.title('K-means')
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.legend(['A', 'B'])
plt.show()
