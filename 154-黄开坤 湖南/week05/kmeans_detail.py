#coding:utf-8

import numpy as np
import math
import matplotlib.pyplot as plt

#欧氏距离
def get_distance(p1, p2):   #直接向量减方便
    temp = 0
    for i in range(len(p1)):  #二维
        temp = temp + pow((p2[i]-p1[i]), 2)
    dis = math.sqrt(temp)
    return dis
# def get_distance(p1, p2):
#     d = math.sqrt(pow((p2[0]-p1[0]), 2) + pow((p2[1]-p1[1]), 2))
#     return d

def calc_center_points(list):
    #计算每列的均值
    means = np.array(list).mean(axis=0)
    # means = list(means)
    # print('means:',means)
    return means

# 检查两个点是否有差别
def check_center_diff(center, new_center):
    n = len(center)
    for c, nc in zip(center, new_center):
        if c != nc:
            return False
    return True


def K_means(points, center_points):
    N = len(points)  # 样本个数
    n = len(points[0])  # 单个样本的维度
    k = len(center_points)  # k个数

    tot = 0
    while True:  # 迭代
        temp_center_points = []  # 记录中心点
        clusters = []  # 记录聚类的结果
        for i in range(k):
            clusters.append([])  # 初始化。[[], [], [], ...]
        # 针对每个点寻找距离最近的中心点
        for i, data in enumerate(points):  # 用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            distances = []
            for c in center_points:
                d = get_distance(data, c)  # 求出两点的距离
                distances.append(d)
            index = distances.index(min(distances))  # 获取最小距离的那个中心点的索引，这个索引对应相应的簇
            clusters[index].append(data)  # 根据这个索引把这个点放到相应的簇中

        tot += 1
        print(tot, '此迭代', clusters)
        k = len(clusters)
        print('K:', k)
        # colors = ['r.', 'g.', 'b.', 'k.', 'y.']  # 颜色和点的样式
        # for i, cluster in enumerate(clusters):
        #     data = np.array(cluster)
        #     data_x = [x[0] for x in data]
        #     data_y = [y[0] for y in data]
        #     plt.figure()
        #     plt.subplot(2, 3, tot)
        #     plt.plot(data_x, data_y, colors[i])
        #     plt.axis([0, 10, 0, 10])

        # 重新计算中心点
        for cluster in clusters:
            # print('cluster:', cluster)
            temp_center_points.append(list(calc_center_points(cluster)))
        print('新中心点：', temp_center_points)
        # 计算中心点时需要将原来的中心点放进去
        for j in range(k):
            if len(clusters[j]) == 0:
                temp_center_points[j] = center_points[j]
        # 判断中心点是否发生变化：即，判断聚类前后样本的类别是否发生变化
        # 判断中心点是否发生变化：即，判断聚类前后样本的类别是否发生变化
        for c, nc in zip(center_points, temp_center_points):
            if not check_center_diff(c, nc):
                center_points = temp_center_points[:]  # 复制一份
                break
        else:  # 如果没有变化，那么退出迭代，聚类结束
            break
    return clusters  # 返回聚类的结果

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

if __name__ == '__main__':
    # points = [[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.], [4.5, 5.], [3., 5.], [1.2, 3.], [8., 7.6]]
    # center_points = [[3., 4.], [7., 8.], [1.2, 3.]]
    x1 = [[0.1956, 0.4280],[0.1061, 0.5523],[0.1276, 0.5703]]
    clusters = K_means(X, x1)
    # clusters = K_means(points, center_points)
    print('clusters[i]:\n', clusters[0],'\n',clusters[1],'\n',clusters[2])
    print('len:', len(clusters))
    print('-----------------------')

    # 画图
    for i in range(len(clusters)):
        marker =  ['x', '*', 'o']
        cl = np.array((clusters[i]))    #二维数组转换成array才可以取整行，整列。
        print('clusters[%d]' % i, cl)
        print('-------')
        plt.scatter(x=cl[:,0], y=cl[:,1], marker=marker[i])
    plt.title('K-means')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(('A', 'B', 'C'))
    plt.show()
    for i, cluster in enumerate(clusters):
        print('cluster:', i, cluster)