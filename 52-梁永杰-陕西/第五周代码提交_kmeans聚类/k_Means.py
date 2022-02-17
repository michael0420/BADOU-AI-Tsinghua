'''
@Auther：Jelly
@用途：用于实现K-Means聚类方法
    相关接口：
        def kmeans_self(dataSet,k)
            函数用途：k-means实现
            函数参数：
                dataSet: 待机算数据
                k: 质心个数
            返回值：
                centroids：新的质心坐标
                cluster：聚类后的个簇坐标
                minDistIndices：聚类后的坐标分组
'''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def calcDis(dataSet,centroids,k):
    '''
    函数用途：计算欧式距离
    函数参数：
        dataSet: 待机算数据
        centroids: 质心坐标
        k: 质心个数
    返回值:
        clalsit: 每个点到质心的距离
    '''

    clalist = []
    for data in dataSet:
        #(np.tile(a, (2, 1))就是把a先沿x轴复制1倍，即没有复制，仍然是[0, 1, 2]。
        # 再把结果沿y方向复制2倍得到array([[0, 1, 2], [0, 1, 2]]))
        diff = np.tile(data,(k,1)) - centroids   #得到与质点的相对距离
        squaredDiff = diff ** 2                  #将所有原素进行平方
        squaredDist = np.sum(squaredDiff,axis=1)  # axis = 1 表示按行进行加和操作，及x^2+y^2
        distance = squaredDist ** 0.5
        clalist.append(distance)   #将结果放入新的数组中储存
    clalist = np.array(clalist) #返回一个点到质心距离的len(dataSet)*k的数组
    return clalist


def classify(dataSet,centroids,k):
    '''
    函数用途：计算质心
    函数参数：
        dataSet: 待机算数据
        centroids: 质心坐标
        k: 质心个数
    返回值：
        changed：改动大小
        newCentriods：新的质心坐标
    '''
    clalist = calcDis(dataSet,centroids,k)
    minDistIndices = np.argmin(clalist,axis=1)  #axis=1 表示求出每行的最小值的下标
    newCentriods = pd.DataFrame(dataSet).groupby(minDistIndices).mean() #groupby(min)按照min（及分组点）进行统计分类
    newCentriods = newCentriods.values

    changed = newCentriods - centroids

    return changed,newCentriods


def kmeans_self(dataSet,k):
    '''
    函数用途：k-means实现
    函数参数：
        dataSet: 待机算数据
        k: 质心个数
    返回值：
        centroids：新的质心坐标
        cluster：聚类后的个簇坐标
        minDistIndices：聚类后的坐标分组
    '''
    #随机生成质心
    centroids = random.sample(dataSet,k)

    #更新质心 直到变化量为零 ，作为一个停止迭代的标志
    changed,newCentroids = classify(dataSet,centroids,k)
    while np.any(changed != 0):
        changed,newCentroids = classify(dataSet,newCentroids,k)

    centroids = sorted(newCentroids.tolist())

    cluster = []
    #根据质心计算每个集群
    clalist = calcDis(dataSet,centroids,k)

    minDistIndices = np.argmin(clalist,axis=1)  # 行比较
    for i in range(k):
        cluster.append([])
    for i,j in enumerate(minDistIndices):    #enumerate()返回为一个从0开始的序号和min里的数据（分组号）
        cluster[j].append(dataSet[i])

    return centroids,cluster,minDistIndices

def createDataSet():
    return [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]



if __name__=='__main__':
    dataset = createDataSet()
    centroids, cluster,labels =  kmeans_self(dataset, 2)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0],dataset[i][1], marker = 'o',color = 'green', s = 40 ,label = '原始点')
                                                  #  记号形状       颜色      点的大小      设置标签
        for j in range(len(centroids)):
            plt.scatter(centroids[j][0],centroids[j][1],marker='x',color='red',s=50,label='质心')
    plt.show()

    img = cv2.imread('lenna.png', 0)
    rows, cols = img.shape[:]
    # 图像转换为一维数据
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)


    centroids, cluster,labels = kmeans_self(list(data),4)

    # 生成最终图像
    labels = np.array(labels)
    dst = labels.reshape((img.shape[0], img.shape[1]))

    # 显示图像
    titles = [u'原始图像', u'聚类图像']
    images = [img, dst]

    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
