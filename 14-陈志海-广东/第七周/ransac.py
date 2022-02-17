"""
@author: 14+陈志海+广东
@fcn：简单实现用ransac回归线性模型y=k*x
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl


def get_shuffle_index(n_sample, n_shuffle):
    """
    :param n_sample: 完整数据集的个数
    :param n_shuffle: 用于打乱后随机抽取的数据个数
    :return:
    idxs_shuffle: 打乱后提取的下标
    idxs_other: 剩余下标
    """
    idxs = np.arange(n_sample)
    np.random.shuffle(idxs)
    idxs_shuffle = idxs[:n_shuffle]
    idxs_other = idxs[n_shuffle:]
    return idxs_shuffle, idxs_other


class Sample:
    """生成理想信号，含高斯噪声信号， 带外群点信号"""
    def __init__(self, n_input, n_outlines):
        self.n_input = n_input
        self.n_outlines = n_outlines
        self.x = 20 * np.random.random(size=n_input)
        self.k = np.random.normal(loc=60, scale=1, size=n_input)
        self.y = self.k * self.x

    def sample_perfect(self):
        data_perfect = np.zeros(shape=(self.n_input, 2))
        data_perfect[:, 0] = self.x
        data_perfect[:, 1] = self.y
        return data_perfect

    def sample_noise(self):
        data_noise = np.zeros(shape=(self.n_input, 2))
        x_noise = self.x + np.random.normal(size=self.n_input)
        y_noise = self.y + np.random.normal(size=self.n_input)
        data_noise[:, 0] = x_noise
        data_noise[:, 1] = y_noise
        return data_noise

    def sample_outlines(self):
        data_outlines = self.sample_noise()
        idxs_shuffle, __ = get_shuffle_index(self.n_input, self.n_outlines)
        data_outlines[idxs_shuffle, 1] = 50 * np.random.random(self.n_outlines)
        return data_outlines


def my_lstsq(data):
    """
    最小二乘法
    :param data: m*2格式数组，第一列为x，第二列为y
    :return:
        p: 拟合多项式系数矩阵
    """
    x = data[:, 0][:, np.newaxis]
    y = data[:, 1][:, np.newaxis]
    # p: parameters, 拟合多项式系数y= p0+p1*x+p2*x^2...,
    # 但是此处x只有1列，因此不存在x^0的系数，所以y=p0*x
    p, __, __, __ = sl.lstsq(x, y)
    return p


def inliners_update(p, idxs_inliners, idxs_test, data_test, criterion):
    """
    更新内群点下标
    :param p: 多项式系数矩阵
    :param idxs_inliners: 原来的内群点下标
    :param idxs_test: 测试数据集下标
    :param data_test: 测试数据集，m*2数组
    :param criterion: 误差阈值
    :return:
        idxs_inliners_updated：更新后的内群点下标
    """
    x = data_test[:, 0]
    y = data_test[:, 1]
    err_points = np.power(p[0] * x - y, 2)       # 测试数据y值与拟合值的误差
    idxs_inliners_add = idxs_test[err_points < criterion]       # 误差小于阈值的下标
    idxs_inliners_updated = np.concatenate((idxs_inliners, idxs_inliners_add))
    return idxs_inliners_updated


def ransac(data,  n_ransac, iteration, criterion):
    """
    ransac回归线性模型y=k*x
    :param data:
    :param n_ransac: 每次随机的内群点个数
    :param iteration: 迭代次数
    :param criterion: 测试集属于内群点的误差阈值
    :return:
        p_result: ransac回归得出的多项式系数矩阵
    """
    n_result = 0                # 最终内群点数量
    idxs_result = None       # 最终内群点的下标
    for i in range(iteration):
        idxs_shuffle, idxs_test = get_shuffle_index(data.shape[0], n_ransac)
        data_ransac = data[idxs_shuffle, :]     # 随机点集用于生成模型
        data_test = data[idxs_test, :]        # 剩余点集用于测试模型
        p = my_lstsq(data_ransac)               # 最小二乘法生成的模型

        # 将符合阈值的测试数据的下标也归为内群点，并更新内群点下标
        idxs_inliners_updated = inliners_update(p, idxs_shuffle, idxs_test, data_test, criterion)
        if idxs_inliners_updated.shape[0] > n_result:
            n_result = idxs_inliners_updated.shape[0]
            idxs_result = idxs_inliners_updated
        iteration += 1

    # 用最终的下标重新计算模型
    data_ransac = data[idxs_result, :]
    p_result = my_lstsq(data_ransac)  # 最小二乘法生成的模型
    return p_result


# main
sample = Sample(n_input=500, n_outlines=100)
data_outlines = sample.sample_outlines()

p_lstsq = my_lstsq(data_outlines)
p_ransac = ransac(data=data_outlines,  n_ransac=5, iteration=100, criterion=5)


plt.figure()
plt.scatter(data_outlines[:, 0], data_outlines[:, 1], s=6, c='k', label="sample")
plt.plot(sample.x, 60 * sample.x, c='r', label="y=60*x")
plt.plot(sample.x, p_lstsq[0] * sample.x, c='b', label=("lstsq k=%.3f" % p_lstsq[0]))
plt.plot(sample.x, p_ransac[0] * sample.x, c='g', label=("ransac k=%.3f" % p_ransac[0]))
plt.legend()
plt.show()

