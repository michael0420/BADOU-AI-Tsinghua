#!/usr/bin/env python 
# coding:utf-8
import os
import random
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt


# 选择质心
def select_centroid(data, n_clusters):
    # 从数据样本中随机选择n_clusters个质心
    copy_data = data.copy()
    centroid_list = []
    for i in range(n_clusters):
        choice_data = random.choice(copy_data)
        copy_data.remove(choice_data)
        centroid_list.append(choice_data)
    if n_clusters != len(centroid_list):
        print("无法选择满足条件的质心数量")
        os.exit(0)
    return centroid_list


# 计算其他元素到每个质心的距离,并归类
def calculate_distance(data, centroid_list):
    # 原始数据,去掉是质心的元素
    copy_data = data.copy()
    for i in range(len(data)):
        if data[i] in centroid_list:
            copy_data.remove(data[i])
    # 将随机得到的初始质心分类
    group_dict = {}
    for i in range(len(centroid_list)):
        group_dict[i] = [centroid_list[i]]
    for i in range(len(copy_data)):
        cnt = 0
        min_dist = 0.0
        element_list = []
        category = 0
        for key, j in group_dict.items():
            # 计算欧式距离
            data_x = copy_data[i][0]
            data_y = copy_data[i][1]
            centroid_list_x = j[0][0]
            centroid_list_y = j[0][1]
            distance = sqrt(pow((data_x - centroid_list_x), 2) + pow((data_y - centroid_list_y), 2))
            # 把第一次循环得到的值当做最小值
            if cnt == 0:
                min_dist = distance
                element_list = copy_data[i]
                category = key
                cnt += 1
            # 当前的最小值和本次计算出的距离进行比较,如果比最小值小,把当前值赋值给最小值
            if min_dist > distance:
                min_dist = distance
                element_list = copy_data[i]
                category = key
        # 将离得最近的点,进行归类
        group_dict.get(category).append(element_list)
    # 遍历字典,将虚拟质心剔除
    for key, value in group_dict.items():
        for item in value:
            if item not in data:
                value.remove(item)
    # 选择新的质心
    new_centroid_list = []
    for key, value in group_dict.items():
        # 每簇的平均数作为新的质心
        new_centroid_list.append(np.mean(np.array(value), axis=0).tolist())
    return new_centroid_list, group_dict


# 多次迭代,直到满足条件或聚类结果不再变化为止
def iterative_calculation(data, centroid_list, criteria):
    num = 1
    while True:
        new_centroid_list, group_dict = calculate_distance(data, centroid_list)
        # 前后两次质心一样,表示聚类完成,即可停止迭代,或者迭代到阈值,即使质心不一样,依旧停止
        if (centroid_list == new_centroid_list) or (num > criteria):
            break
        centroid_list = new_centroid_list
        num += 1
        print("迭代次数: \n", num)
    return group_dict


if __name__ == '__main__':
    # 样本数据

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

# 分为3个类簇
n_clusters = 3
# 最大迭代次数
criteria = 20
centroid_list = select_centroid(X, n_clusters)
group_dict = iterative_calculation(X, centroid_list, criteria)
color_list = ["red", "green", "blue"]
for key, value in group_dict.items():
    x = [v[0] for v in value]
    y = [v[1] for v in value]
    plt.scatter(x, y, c=color_list[key], marker="o")
plt.title("聚类结果")
# 显示中文
plt.rcParams["font.sans-serif"] = ['Arial Unicode MS']
plt.show()
