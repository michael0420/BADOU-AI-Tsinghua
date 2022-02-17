#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2022/01/09 12:18:35
@Author : LuZhanglin

K-means 聚类计算
'''

import numpy as np
import random


class Kmeans:
    """
    Args:
        K (int) : K个类
        data (np.array) : 数据 shape(num_points, num_dims)
    """
    def __init__(self, K, data) -> None:
        self.K = K
        self.data = data

    def loop(self):
        """选择K个点作为中心"""
        num_points, num_dims = self.data.shape

        rand_idx = random.sample(range(num_points), self.K)
        center_p = self.data[rand_idx]
        # 其他点
        res_points_idx = list(set(range(num_points)) - set(rand_idx))
        res_points = self.data[res_points_idx]

        # 迭代直到聚类结果不再变化
        iter_step = 0
        early_stop = 2000
        flag = 0
        while True:
            iter_step += 1
            if iter_step > 1:
                res_points = self.data
            dist_ls = []
            for p in center_p:
                # 计算其余点分别到中心点的距离  shape (len(res_points))
                dist = np.sqrt(((res_points - p) ** 2).sum(axis=1))
                dist_ls.append(dist)
            dist_sum = np.min(dist_ls, axis=0).sum()
            if iter_step > 1:
                # 一定时间后距离不再减小则停止迭代
                if flag == early_stop:
                    break
                if (temp_dist <= dist_sum):
                    flag += 1
            temp_dist = dist_sum
            # dist_ls shape (num_center, num_other_points)
            clusterd_idx = np.argmin(dist_ls, axis=0)
            if iter_step == 1:
                # 根据距离划分K类
                k_groups = [[center_p[i].tolist()] for i in range(self.K)]
            else:
                k_groups = [[] for i in range(self.K)]

            center_p = np.empty((self.K, num_dims))
            for i in range(self.K):
                k_groups[i] += res_points[clusterd_idx == i].tolist()
                # 更新类质心
                center_p[i] = np.mean(k_groups[i], axis=0)
        return k_groups


if __name__ == "__main__":
    data = np.array([[0,0],
                     [1,2],
                     [3,1],
                     [8,8],
                     [9,10],
                     [10,7]])
    
    X = np.array([[0.0888, 0.5885],
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
                ])
    
    K = 3
    # cluster = Kmeans(K, data)
    cluster = Kmeans(K, X)
    result = cluster.loop()
    import matplotlib.pyplot as plt
    cs = ['r', 'g', 'b', 'o', 'gray']
    for i in range(K):
        xy = np.array(result[i])
        plt.scatter(xy[:, 0], xy[:, 1], c=cs[i])
    plt.show()


