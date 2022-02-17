import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def create_random_centers(data, K):
    data = np.array(data)
    rows, cols = data.shape
    max_min = np.zeros((2, cols))     # data的各列的最大最小值, 第一行为max, 第二行为min
    centers = np.zeros((K, cols))     # K个中心点
    max_min[0, :] = np.max(data, axis=0)
    max_min[1, :] = np.min(data, axis=0)

    # 生成K个随机中心点
    for col in range(cols):
        _max = max_min[0, col]
        _min = max_min[1, col]
        for row in range(K):
            centers[row, col] = random.uniform(_min, _max)

    return centers


def sum_new_center(clusters, cur_centers):
    # 根据分类的K个类簇，重新计算每个类簇的数据中心
    K = len(clusters)
    rows, cols = cur_centers.shape
    centers = np.zeros((rows, cols))
    for i, cluster in enumerate(clusters):
        if cluster:         # 如果类簇不为空，则直接计算新质心
            centers[i] = np.mean(cluster, axis=0)
        else:               # 如果类簇为空，则用当前所有其他质心的均值作为新质心
            tmp = np.delete(centers, i, axis=0)
            centers[i] = np.mean(tmp, axis=0)

    return centers


def sum_distance(centers, point):
    """
    计算点到各中心点的距离, 及离point最近的center的索引
    :param centers: 中心点集
    :param point:
    :return: distance：point离所有中心点的距离；idx_max: 离point最近的center的索引
    """
    div = np.power(centers - point, 2)      # 数组[δx^2, δy^2]
    distance = np.sqrt(np.sum(div, axis=1))
    idx_max = np.argmax(-distance)
    return distance, idx_max


def if_stop(criteria, old_centers, cur_centers, iter):
    # 判断是否停止迭代
    stop_mode = criteria[0]
    iters = criteria[1]
    eps = criteria[2]
    mse = np.sqrt(np.sum(np.power(cur_centers - old_centers, 2)))
    stop_flag = False
    if stop_mode == "EPS":
        if mse <= eps:
            stop_flag = True
    elif stop_mode == "ITER":
        if iter >= iters:
            stop_flag = True
    elif stop_mode == "EPS+ITER":
        if mse <= eps or iter >= iters:
            stop_flag = True

    return stop_flag, mse


def kmeans(data, K, bestLabels, criteria, attempts=None, flags=None, centers=None):
    """
    :param data: 数据集
    :param K: 聚类类别个数
    :param bestLabels: 类别标签
    :param criteria: 迭代停止模式。0-EPS；1-迭代次数；2-EPS+迭代次数
    :param attempts: 重复试验迭代的次数，返回最佳的一次
    :param flags: 初始中心选择标志
    :param centers: 由聚类中心组成的数组
    :return:
    clusters: 各类簇的元素值
    ids： 各类簇的元素对应原数据的位置
    centers: 各类簇的中心坐标
    """
    assert K > 1 and len(data)
    if not centers:
        centers = create_random_centers(data, K)

    iter = 1
    while True:
        clusters = [[] for i in range(K)]  # 生成分类簇用于存放点集
        ids = np.zeros(len(data), dtype=np.uint8)  # data中，每个元素属于第几个类簇
        time1 = time.time()
        for i in range(len(data)):
            point = data[i]
            distance, id_max = sum_distance(centers, point)         # 存放该点距离K个中心的距离
            clusters[id_max].append(point)                          # 将离质心最近的点放到质心对应的类簇
            ids[i] = id_max
        time2 = time.time()
        cost = time2 - time1
        cur_centers = sum_new_center(clusters, cur_centers=centers)     # 根据类簇计算新质心
        stop_flag, mse = if_stop(criteria, centers, cur_centers, iter)
        print("iter: %d, mse: %f cost time: %.1f" % (iter, mse, cost))
        if stop_flag:
            break
        centers = cur_centers
        iter += 1
    ids = np.array(ids, dtype=np.uint8)
    return clusters, ids, cur_centers


def fcn_kmeans_data():
    # 示例1：对数组kmeans
    criteria = ("EPS", 2000, 0.00001)
    K = 3
    c = ('r', 'g', 'b', 'y', 'k', 'c')
    data = [[0.0888, 0.5885],
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
    clusters, ids, cur_centers = kmeans(data, K, None, criteria, None, None)
    # figure
    plt.figure()
    data = np.array(data)
    plt.axis([0, 0.3, 0.3, 1])
    for i, cluster in enumerate(clusters):
        if cluster:
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1], c=c[i], marker='x', linewidths=4)
        plt.scatter(cur_centers[i, 0], cur_centers[i, 1], c=c[i], marker=10, linewidths=4)
    plt.axis([0, 0.3, 0.3, 0.9])
    plt.show()


def fcn_kmeans_image():
    # 示例2：对RGB图像kmeans
    img_bgr = cv2.imread("lenna.png", -1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.title("lenna")
    plt.axis("off")

    img_reshape = np.reshape(img_rgb, (-1, 3))
    criteria = ("EPS+ITER", 5, 2)
    clusters, ids, cur_centers = kmeans(img_reshape, 3, None, criteria, None, None)

    # 将图像的像素替换为centers值
    cur_centers = np.floor(cur_centers)
    ids = np.array(ids)
    for i in range(ids.shape[0]):
        img_reshape[i] = cur_centers[ids[i]]
    img_kmeans = np.reshape(img_reshape, img_bgr.shape)

    plt.subplot(122)
    plt.imshow(img_kmeans)
    plt.title("lenna_kmeans")
    plt.axis("off")
    plt.show()


# main
# 对数字进行kmeans处理
print("实例1：处理简单数组")
fcn_kmeans_data()
# 对RGB图像进行kmeans处理，处理时间要十几秒，待优化
print("实例2：处理lenna图片")
fcn_kmeans_image()
