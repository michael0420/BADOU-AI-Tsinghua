import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
数据集X，二维矩阵，第一列为球员每分钟助攻数，第二列为球员每分钟的分数
'''
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1],

     [4, 4],
     [4, 5],
     [5, 4],
     [5, 5],

     [0, 4],
     [0, 5],
     [1, 4],
     [1, 5],

     [4, 0],
     [4, 1],
     [5, 0],
     [5, 1]]


# 计算欧氏距离
def Euclidistance(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))

# 构建聚簇中心，去K个随机质心
'''
def randCent(dataset, k):
    n = dataset.shape[1]  # 样本维度
    centroids = np.mat(np.zeros((k, n))) # K个随机质心矩阵，保存质心
    for j in range(n):
        minJ = np.min(dataset[:, j])  # 遍历所有样本的每一个维度，也就是每一列，找出最小的值
        maxJ = np.max(dataset[:, j])  # 找出最大的值
        rangeJ = maxJ - minJ # 得到范围
        randvalue = np.random.rand(k, 1) # 返回一组服从0~1的随机样本矩阵，(K,1)为得到的随机样本矩阵大小，比如(4,1)表示得到一个
                                         # 4行1列的随机样本矩阵
        centroids[:, j] = minJ + rangeJ * randvalue
    return centroids
'''
def randCent(dataset, k):
    m, n = dataset.shape
    centroids = np.zeros((k, n)) # K个随机质心矩阵，保存质心
    index = []
    for i in range(k):
        value = int(np.random.uniform(0, m)) # 在0~m之间取随机数，也就是随机挑选质心

        # 判断新取的随机值是否和以前的值一样，一样的话要重新取值
        for j in range(len(index)):
            while value == index[j]:
                value = int(np.random.uniform(0, m)) # 重新取值，直到不相等
        index.append(value)  # 将得到的随机值添加到index
        centroids[i, :] = dataset[value, :]

    return centroids

# K均值聚类基础算法，后续会在此基础上实现二分K-Means算法
def Kmeans(dataset, k):
    m = dataset.shape[0]
    clusterAssment = np.mat(np.zeros((m, 2))) # 第一列存放该样本所属的质心，第二列存放每个样本到质心的距离
    centroids = randCent(dataset, k) # 得到初始质心
    clusterChanged = True # 用来判断聚类是否已经收敛

    while clusterChanged:
        clusterChanged = False

        # 1、遍历所有的样本
        for i in range(m):  #　处理每个样本点，将其划分到离得最近的质心
            minDist = np.inf # 最小距离，默认为﹢∞
            minIndex = -1    # 纪录最小距离样本索引,也就是样本离哪个质心近

            # 2、遍历所有的质心，找出样本距离最近的质心，
            for j in range(k):
                distance = Euclidistance(centroids[j, :], dataset[i, :]) # 计算距离近
                if distance < minDist:
                    minDist = distance
                    minIndex = j

            # 3、更新每一个行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True   # 标记簇修改过

            clusterAssment[i, 0] = minIndex
            clusterAssment[i, 1] = minDist**2 # SSE，均方误差

        # 4、聚类完成以后，更新质心
        for j in range(k):
            pointsCluster = dataset[np.nonzero(clusterAssment[:, 0].A == j)[0]] # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsCluster, axis=0) # 对列求平均值

    # 5、提取出标签
    labels = []
    for i in range(m):
        index = int(clusterAssment[i, 0])
        labels.append(index)

    return centroids, labels, clusterAssment # 返回质心和聚类结果



'''
二分法K-Means均值聚类，在基础的K-Means算法基础上升级而来
避免陷入局部最小值，而非全局最小值
算法步骤: 
1.将所有点看成一个簇，求质心
2.当簇数目小于k时
    对于每一个簇
        计算总误差
        在给定的簇上面进行k-均值聚类（k=2），即一分为2
        计算将该簇一分为二后的总误差
    选择使得误差最小的那个簇进行划分操作
'''
'''
def bi_Kmeans(dataset, k, distCompute=Euclidistance):
    m, n = np.shape(dataset)
    clusterAssment = np.mat(np.zeros((m, 2))) # 第一列存放该样本所属的质心，第二列存放每个样本到质心的距离

    # 1、创建一个初始簇，也就是全部点看成一个簇
    centroid0 = np.mean(dataset, axis=0).tolist()[0]
    centList = [centroid0] # 1个质心


    # 计算初始SSE，误差平方和
    for i in range(m):
        clusterAssment[i, 1] = Euclidistance(np.mat(centroid0), dataset[i, :])**2 # 计算距离近

    while len(centList) < k:
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataset[np.nonzero(clusterAssment[:, 0].A == i)[0], :] # 获取当前的簇
            centroidMat, labels, splitClustAss = Kmeans(np.array(ptsInCurrCluster), 2) # K=2
            sseSplit = np.sum(splitClustAss[:, 1]) # 计算误差均方和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            # print("sseSplit, sseNotSplit = ", sseSplit, sseNotSplit)

            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # print("the bestCenToSplit is:", bestCentToSplit)
        # print("the len of bestClustAss is:", len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        print("len of centList is", len(centList))
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss


    return np.mat(centList), clusterAssment
'''

# 绘制图形
def draw(dataset, centroids, clusterAssment, k):
    dataset = np.array(dataset)
    m, n = dataset.shape
    colors = ['r', 'g', 'b', 'yellow', 'black', 'pink']

    plt.figure(1)

    # 绘制所有样本
    for i in range(m):
        index = int(clusterAssment[i, 0])
        plt.scatter(x=dataset[i, 0], y=dataset[i, 1], c=colors[index])

    # 绘制质心
    for i in range(k):
        plt.scatter(x=centroids[i, 0], y=centroids[i, 1], marker='*', c=colors[i])

    plt.show()


if __name__ == '__main__':
    X = np.array(X)
    centroids, labels, clusterAssment = Kmeans(X, 4)
    draw(X, centroids, clusterAssment, 4)

    # 1、读取图像
    '''
    img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
    print(img.shape)

    img_rows = img.shape[0]  # 图像行数
    img_clos = img.shape[1]  # 图像列数
    '''


