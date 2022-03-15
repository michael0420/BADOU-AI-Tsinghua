# 目标：手写实现kmeans的详细算法，
# 添加接口调用canny官方接口或自制接口进行聚类

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

'''
kmeans 接口选择，1：自制接口; 2：官方接口;
'''
kmeans_type = 1
# kmeans_type = 2

"""
原始输入数据集
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
     [0.1956, 0.4280]
     ]


def show(src):
    # 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列，分别存储xy轴坐标
    x = [n[0] for n in X]
    # print(x)
    y = [n[1] for n in X]
    # print(y)

    ''' 
     绘制散点图 
     参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
     '''
    # plt.scatter(x, y, c=src, marker='x')
    plt.scatter(x, y, c=src, marker='o')

    # 绘制标题
    plt.title("Kmeans Basketball Data")

    # 绘制x轴和y轴坐标
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")

    # 设置右上角图例
    plt.legend(["A", "B", "C"])

    # 显示图形
    plt.show()


def kmeans(data, k, stop_cnt=-1):
    """
    :param d: 输入数据，类型 ndarray
    :param k: k类
    :param stop_cnt: 停止遍历次数，达到stop_cnt则停止，如果为-1，则运行到不变为止
    :return: 参数的分类结果
    """
    # 转换原始数据输入类型
    d = np.array(data)
    # 随机获取k个随机数
    ll = d.shape[0]
    rarr = np.random.random(k)  # 随机获取0~1
    rarr = np.floor((rarr * ll)).astype(int)  # 乘数据集大小得到随机点坐标,flooor 向下整确保能取到0
    # print(rarr)
    # 记录初始中心点
    center = d[rarr]
    print("初始center\n", center)
    print('-' * 20)
    # 初始化类别数组
    cls = np.zeros(ll, np.int32)
    f = True
    cnt = 0
    while f:
        cnt += 1
        for i in range(ll):
            # 获取第i组数据到k个center的距离的平方和
            n = np.sum(np.square(d[i] - center), axis=1)  # k个(dx^2+dy+2) axis=1，按行取整
            cls[i] = np.argmin(n)  # 取最小的中心点标号，作为标记
        print(cls)
        f = False
        # 判断退出条件
        # 1、达到指定运行次数
        if stop_cnt > 0:
            if cnt > stop_cnt:
                break
        # 2、两次平均值接近,退出，否则继续
        # 计算新的中心点
        for i in range(k):
            # 获取样本中类别为i的集合
            c = d[cls == i]
            # print(c)
            newcenter = np.mean(c, axis=0)  # 按列计算所有样本中的点的平均值，得到新的中心点坐标
            # print(newcenter)
            # 获取两次中心点的距离差
            dc = np.sqrt(np.sum(np.square(center[i] - newcenter)))
            print(dc)
            center[i] = newcenter
            if dc > 1e-10 or f:
                center[i] = newcenter
                f = True
        print(center)
        print('-' * 20)
    print(cls)
    return cls


y_pred = np.zeros((1, 1))

if kmeans_type == 1:
    k = 3
    y_pred = kmeans(X, k)
else:
    """
    sklearn 中
    n_clusters : 聚类的个数k，default：8.
    init : 初始化的方式，default：k-means++
    n_init : 运行k-means的次数，最后取效果最好的一次, 默认值: 10
    max_iter : 最大迭代次数, default: 300
    tol : 收敛的阈值, default: 1e-4
    n_jobs : 多线程运算, default=None，None代表一个线程，-1代表启用计算机的全部线程。
    algorithm : 有“auto”, “full” or “elkan”三种选择。"full"就是我们传统的K-Means算法， “elkan”是我们讲的elkan K-Means算法。默认的"auto"则会根据数据值是否是稀疏的，来决定如何选择"full"和“elkan”。一般数据是稠密的，那么就是“elkan”，否则就是"full"。一般来说建议直接用默认的"auto"。
    """
    k = 3
    clf = KMeans(n_clusters=3)
    y_pred = clf.fit_predict(X)

if type(y_pred) is np.ndarray:
    if y_pred.shape[0] == len(X):
        # 绘制散点图
        show(y_pred)
    else:
        print('y_shape:', y_pred.shape[0])
        print('X_shape:', len(X))
        print("err")
