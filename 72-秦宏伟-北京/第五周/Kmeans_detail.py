#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

"""
Kmeans详细的聚类实现
df
    输入的数据集,M*N的二维数组
K
    要聚成K类
"""
def kmeans(df,K):
    preResult = np.zeros(len(df))
    M = len(df[0])
    N = len(df)
    P = np.empty(shape=(K,M)) #质心
    #初始化P
    for i in range(K):
        for j in range(M):
            P[i][j] = df[i][j]
    #判断所有点分别到质心的距离，并获得分类结果
    disResult = getSortedDistance(df,P)
    while True:
        if (preResult==disResult).all():
            #循环结束
            return disResult
        else:
            preResult = disResult
            #获取新的质心
            P = getP(df,disResult,K)
            disResult = getSortedDistance(df, P)

"""
根据聚类的结果，获取每个聚类的数据集
"""
def getClusterResult(df,disResult,K):
    listResult = []
    for i in range(K):
        listResult.append([])
    for i in range(len(disResult)):
        listResult[int(disResult[i])].append(df[i])
    return listResult
"""
获取新的质心
原始数据df
上一轮聚类结果disResult
"""
def getP(df,disResult,K):
    P = np.empty(shape=(K, len(df[0])))  # 质心
    # G = []
    listResult = []
    for i in range(K):
        listResult.append([])
    for i in range(len(disResult)):
        listResult[int(disResult[i])].append(df[i])

    for i in range(K):
        X = 0
        Y = 0
        for j in range(len(listResult[i])):
            X += listResult[i][j][0]
            Y += listResult[i][j][1]
        X = X / len(listResult[i])
        Y = Y / len(listResult[i])
        P[i] = [X,Y]
        # G.append([X,Y])
    return P

"""
判断所有点分别到质心的距离,并分类
"""
def getSortedDistance(df,P):
    disResult = np.zeros(len(df),int)
    for i in range(len(df)):
        #分别计算每个点到质心的距离
        # distmp =
        dis = np.zeros(len(P))
        for j in range(len(P)):
            dis[j] = calcDistance(df[i],P[j])
        #获取距离最短的索引，作为类别
        index = np.where(dis == dis.min())
        disResult[i] = index[0][0]
    return disResult

"""
计算每个点point到质心G的距离
"""
def calcDistance(point,G):
    totalDis = 0
    for i in range(len(point)):
        totalDis += (point[i]-G[i])**2
    sqrtDis = math.sqrt(totalDis)
    return sqrtDis

if __name__ == '__main__':
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
    K=3
    y_pred = kmeans(X ,K)

    # 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
    x = [n[0] for n in X]
    print(x)
    y = [n[1] for n in X]
    print(y)

    ''' 
    绘制散点图 
    参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
    '''
    plt.scatter(x, y, c=y_pred, marker='x')

    # 绘制标题
    plt.title("Kmeans-Basketball Data")

    # 绘制x轴和y轴坐标
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")

    # 设置右上角图例
    plt.legend(["A", "B", "C"])

    # 显示图形
    plt.show()