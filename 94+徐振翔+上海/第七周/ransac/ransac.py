# Ransnac思想在详细实现

"""
Ransac实现步骤:
1. 在数据中随机选择几个点设定为内群
2. 计算适合内群的模型 e.g. y=ax+b ->y=2x+3 y=4x+5
3. 把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群 e.g. hi=2xi+3->ri
4. 记下内群数量
5. 重复以上步骤
6. 比较哪次计算中内群数量最多,内群最多的那次所建的模型就是我们所要求的解
可设置迭代次数及迭代阈值
注意：w一致的情况下迭代次数越大效果越好
"""

import numpy as np
import random
import scipy as sp
import scipy.linalg as sl


def prepar_test_data():
    """
    生成数据
    :return:
    x_noisy: 测试数据x坐标
    y_noisy: 测试数据y坐标
    """
    n = 500  # 样本个数
    x_exact = 20 * np.random.random((n, 1))  # 随机生成n个x值
    # 随机生成k与b
    k = 60 * np.random.random()
    b = 10 * np.random.random()
    y_exact = x_exact * k + b  # y = kx+b
    # print(y_exact[1:10])
    # 加入高斯噪声,最小二乘能很好的处理
    x_noisy = x_exact + np.random.random(x_exact.shape)  # 500 * 1行向量,代表Xi
    y_noisy = y_exact + np.random.random(y_exact.shape)  # 500 * 1行向量,代表Yi
    # 添加局外点
    n_out = 100  # 局外点的数量
    all_idx = np.arange(x_noisy.shape[0])  # 获取所有数据的索引
    np.random.shuffle(all_idx)  # 将all_idx打乱
    out_idx = all_idx[:n_out]  # 随机选择n_out个局外点
    # 将部分数据替换为局外点
    x_noisy[out_idx] = 30 * np.random.random((n_out, 1))
    y_noisy[out_idx] = 70 * np.random.random((n_out, 1))
    return x_exact, y_exact, x_noisy, y_noisy, k, b


class LinearLeastSauareModel:
    # 构造最小二乘类
    # 计算模型参数与损失
    def __init__(self):
        pass

    def fit(self, data):
        # 计算模型参数
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        x = np.vstack([data[:, 0]]).T  # 第一列Xi-->行Xi
        y = np.vstack([data[:, 1]]).T  # 第二列Yi-->行Yi
        t, __, __, __ = sl.lstsq(x, y)  # residues:残差和
        return t  # 返回最小平方和向量

    def get_err(self, data, model):
        x = np.vstack([data[:, 0]]).T  # 第一列Xi-->行Xi
        y = np.vstack([data[:, 1]]).T  # 第二列Yi-->行Yi
        y_fit = np.dot(x, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_lst = np.sum((y_fit - y) ** 2, axis=1)  # 误差平方列表
        return err_lst


def ransac(data, model, n, k, t, d, return_all=False):
    """
    :param data: 样本点
    :param model: 假设模型:事先自己确定
    :param n: 生成模型所需的最少样本点
    :param k: 最大迭代次数
    :param t: 阈值:作为判断点满足模型的条件
    :param d: 拟合较好时,需要的样本点最少的个数,当做阈值看待
    :param return_all: 是否返回点集
    :return: 最优拟合解（返回nil,如果未找到）
    """
    cnt = 0  # 迭代次数
    best_fit = None  # 最优解
    best_err = np.inf  # 误差，设置默认值
    # print(best_err)
    best_in_idx = None  # 最优内群点
    while cnt < k:
        all_idx = np.arange(data.shape[0])  # 创建n个data下标索引
        np.random.shuffle(all_idx)  # 打乱下标索引
        maybe_in_idx, test_data_idx = all_idx[:n], all_idx[n:]  # 获取随机下标的两类，一类是可能的内群点的索引，另一类是测试点索引
        maybe_in_data, test_data = data[maybe_in_idx], data[test_data_idx]  # 获取内群点数据及测试数据
        # print(maybe_in_data[:10])
        # print(test_data[:10])
        maybemodel = model.fit(maybe_in_data)  # 获取内群点拟合模型
        # print(k, b, los)
        test_err = model.get_err(test_data, maybemodel)  # 计算每个测试点的误差平方和
        # print(type(test_err))
        # print(test_err)
        new_in_idx = test_data_idx[test_err < t]  # 获取新的内群点数
        # 如果合计内群点数大于阈值则计算拟合效果
        if len(new_in_idx) + n > d:
            new_in_data = data[new_in_idx, :]
            all_in_data = np.concatenate((maybe_in_data, new_in_data))  # 内群点拼接
            all_in_model = model.fit(all_in_data)  # 获取新的拟合模型及平均误差
            all_in_err = model.get_err(all_in_data, all_in_model)
            thiserr = np.mean(all_in_err)  # 平均误差作为新的误差
            if thiserr < best_err:
                best_fit = all_in_model
                best_err = thiserr
                best_in_idx = np.concatenate((maybe_in_idx, new_in_idx))  # 更新局内点,将新点加入
        cnt += 1

    if best_fit is None:
        raise ValueError("未找到最优解")
    if return_all:
        return best_fit, best_in_idx
    else:
        return best_fit


def show(all_data, A_exact, B_exact, A_noisy, B_noisy, ransac_fit, ransac_data, k, b):
    """
    绘图
    """
    import pylab
    sort_idxs = np.argsort(A_exact[:, 0])
    A_sorted = A_exact[sort_idxs]  # 确定的内群点
    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(A_noisy[ransac_data, 0], B_noisy[ransac_data, 0], 'bx', label="RANSAC data")
    pylab.plot(A_sorted[:, 0], np.dot(A_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    pylab.plot(x_exact[:, 0], (k * x_exact + b)[:, 0], label="exact_system")
    # 用最小二乘法直接拟合出结果的线
    linear_fit, __, __, __ = sp.linalg.lstsq(np.vstack([all_data[:, 0]]).T, np.vstack([all_data[:, 1]]).T)
    pylab.plot(A_sorted[:, 0], np.dot(A_sorted, linear_fit)[:, 0], label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    x_exact, y_exact, x_noisy, y_noisy, k_base, b_base = prepar_test_data()
    all_data = np.hstack((x_noisy, y_noisy))  # 以形式([Xi,Yi]....)拼接数据 shape:(500,2)500行2列
    nn = x_noisy.shape[0]

    model = LinearLeastSauareModel()  # 类的实例化:用之前的最小二乘法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, return_all=True)  # ransac迭代,寻找最优解
    # ransac_fit, ransac_data = ransac(all_data, model, 50, 10000, 8e3, 250, return_all=True)  # ransac迭代,寻找最优解

    show(all_data, x_exact, y_exact, x_noisy, y_noisy, ransac_fit, ransac_data, k_base, b_base)

