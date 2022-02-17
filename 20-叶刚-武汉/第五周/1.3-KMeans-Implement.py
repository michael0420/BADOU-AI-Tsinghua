"""
@GiffordY
KMeans算法实现
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, num_clusters: int):
        self.data = np.array(data)
        # 1. 确定K值，即将数据为K簇
        self.num_clusters = np.uint(num_clusters)

    def _check_params(self):
        # n_clusters参数类型和大小检查
        type_flag = isinstance(self.num_clusters, int) or isinstance(self.num_clusters, np.uint)
        if not type_flag:
            raise TypeError(f"The Type of num_clusters={self.num_clusters} should be int.")

        if self.num_clusters <= 0:
            raise ValueError(f"num_clusters={self.num_clusters} should be > 0.")

        if self.data.shape[0] < self.num_clusters:
            raise ValueError(f"n_samples={self.data.shape[0]} should be >= num_clusters={self.num_clusters}.")

    def init_centroids(self):
        # 初始化质心的坐标
        num_samples = self.data.shape[0]
        # num_features = self.000-data.shape[1]
        # centroids = np.zeros((self.num_clusters, num_features), dtype=np.float32)
        # np.random.permutation()函数将一个序列打乱，返回新的乱序序列（原序列顺序不变）
        indices = np.random.permutation(num_samples)  # 打乱序号
        centroids = self.data[indices[:self.num_clusters], :]  # 取乱序后靠前的num_clusters个样本点作为初始质心
        print('init_centroids = ', centroids)
        return centroids

    def init_centroids_random(self):
        if np.max(self.data) > self.num_clusters:
            cxs = np.random.randint(np.min(self.data[:, 0]), np.max(self.data[:, 0]), size=self.num_clusters)
            cys = np.random.randint(np.min(self.data[:, 1]), np.max(self.data[:, 1]), size=self.num_clusters)
        else:  # 随机生成[0, 1)之间的数
            cxs = np.random.random(size=self.num_clusters)
            cys = np.random.random(size=self.num_clusters)
        centroids = np.array(list(zip(cxs, cys)), dtype=np.float32)
        print('init centroids = ', centroids)
        return centroids

    def calc_distance(self, A, B, ord=2, axis=1):
        # ord=2，l2范数；axis=1表示按行向量处理，求多个行向量的范数; axis=None表示矩阵范数
        A = np.array(A)
        B = np.array(B)
        dists = np.linalg.norm(A - B, ord=ord, axis=axis)
        return dists

    def update_centroids(self, predict_labels, centroids):
        num_features = self.data.shape[1]
        new_centroids = np.zeros((self.num_clusters, num_features))
        for k in range(self.num_clusters):  # 对于每一个簇
            same_cluster_points = []
            for i in range(len(predict_labels)):  # 对于每一个样本
                if predict_labels[i] == k:  # 如果样本被划分为第k簇，则将所有样本点放入same_cluster_points
                    same_cluster_points.append(self.data[i])
            same_cluster_points = np.array(same_cluster_points)
            if len(same_cluster_points) != 0:
                new_centroids[k] = np.mean(same_cluster_points, axis=0)  # axis=0，压缩行，对列求均值
            else:
                new_centroids[k] = centroids[k]
        print('new_centroids = ', new_centroids)
        return new_centroids

    def fit(self):
        # 检查参数
        self._check_params()
        num_samples = self.data.shape[0]
        # 2. 初始化每个簇的质心
        centroids = self.init_centroids()
        # 初始化临时变量
        old_centroids = np.zeros(centroids.shape, dtype=np.float32)
        run_flag = np.sum(self.calc_distance(centroids, old_centroids, ord=2, axis=1))
        predict_labels = np.zeros(num_samples, dtype=np.uint)  # 存储每个样本属于哪个簇（label)
        while run_flag:
            # predict_labels = np.zeros(num_samples, dtype=np.uint)     # 存储每个样本属于哪个簇
            # 3.1 计算每个样本点到质心的距离
            for i in range(num_samples):
                dists = self.calc_distance(self.data[i], centroids)
                # print('A={}, dist={}'.format(self.000-data[i], dists))
                # 3.2 将每个样本点划分到离它距离最近的簇
                kind = np.argmin(dists)
                predict_labels[i] = kind
                # print('A={}, dist={}, kind={}'.format(self.000-data[i], dists, kind))

            # 4. 更新质心点。取每个簇的平均坐标值作为该簇的新质心
            old_centroids = deepcopy(centroids)
            centroids = self.update_centroids(predict_labels, centroids)
            run_flag = np.sum(self.calc_distance(centroids, old_centroids, ord=2, axis=1))
        return predict_labels, centroids


if __name__ == '__main__':
    dataset = [[0.0888, 0.5885],
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

    dataset = np.array(dataset)

    k_clusters = 3
    kmeans = KMeans(dataset, k_clusters)
    pred_labels, pred_centroids = kmeans.fit()
    print('pred_labels = ', pred_labels)
    print('pred_centroids = ', pred_centroids)

    # 绘制结果
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for ii in range(k_clusters):
        points = np.array([dataset[jj] for jj in range(len(dataset)) if pred_labels[jj] == ii])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[ii])
    ax.scatter(pred_centroids[:, 0], pred_centroids[:, 1], marker='*', s=100, c='r')
    plt.legend(["A", "B", "C"])
    plt.show()
