#!/usr/bin/env python 
# coding:utf-8

import numpy as np
import pylab


# 生成数据
def create_test_data():
    # 样本数量
    n_sample = 500
    # 随机生成x的值
    x_exact = 20 * np.random.normal(size=(n_sample, 1))
    # 随机斜率
    exact_fit = 60 * np.random.normal(size=(1, 1))
    # y的值
    y_exact = np.dot(x_exact, exact_fit)
    # 加入高斯噪声
    x_noisy = x_exact + np.random.normal(size=(n_sample, 1))
    y_noisy = y_exact + np.random.normal(size=(n_sample, 1))
    # 局外点的数量
    n_out_liers = 100
    # 获取所有数据的索引
    all_idx = np.arange(n_sample)
    # 顺序打乱
    np.random.shuffle(all_idx)
    # 取其部分索引
    out_liers_idx = all_idx[:n_out_liers]
    # 将部分数据替换为局外点
    x_noisy[out_liers_idx] = 30 * np.random.normal(size=(n_out_liers, 1))
    y_noisy[out_liers_idx] = 50 * np.random.normal(size=(n_out_liers, 1))
    # 将x,y拼接成完整数据
    all_data = np.concatenate((x_noisy, y_noisy), axis=1)
    return all_data, x_exact, y_exact


# 切割数据,一部分是内群点,一部分是离散点
def split_data(data, n):
    # 所有数据的索引
    all_data_idx = np.arange(data.shape[0])
    # 打乱顺序
    np.random.shuffle(all_data_idx)
    # 取n个作为群内点
    inliers = data[all_data_idx[:n]]
    other_data = data[all_data_idx[n:]]
    return inliers, other_data


# 构造最小二乘类
class LinearLeastSquareModel:
    # 计算模型参数
    def fit(self, data):
        # 列x 转为行x
        x_noisy = np.vstack([data[:, 0]]).T
        # 列y 转为行y
        y_noisy = np.vstack([data[:, 1]]).T
        # 获取斜率
        slope = np.linalg.lstsq(x_noisy, y_noisy, rcond=-1)[0]
        return slope

    # 求误差
    def get_err(self, data, slope):
        # 列x 转为行x
        x_noisy = np.vstack([data[:, 0]]).T
        # 列y 转为行y
        y_noisy = np.vstack([data[:, 1]]).T
        # 拟合出的y
        y_fit = np.dot(x_noisy, slope)
        # 误差平方列表
        data_err_list = np.sum((y_fit - y_noisy) ** 2, axis=1)
        return data_err_list


# ransac实现
def ransac(data, model, n, k, t, d, return_all=False):
    # 迭代次数
    n_iterations = 0
    # 最优解
    best_fix = None
    # 误差
    best_err = np.inf
    # 最优内群点
    best_inliers = None
    while n_iterations < k:
        # 切割数据
        init_inliers, other_data = split_data(data, n)
        # 用内群点去拟合模型
        init_fit = model.fit(init_inliers)
        # 计算误差
        data_err_list = model.get_err(other_data, init_fit)
        # 从另一部分数据计算出符合阈值的新内群点
        new_inliers = other_data[data_err_list < t]
        if len(new_inliers) > d:
            # 将新旧内群点进行连接
            all_inliers = np.concatenate((init_inliers, new_inliers))
            # 再次拟合模型
            better_fit = model.fit(all_inliers)
            better_err_list = model.get_err(all_inliers, better_fit)
            # 平均误差作为新误差
            new_err = np.mean(better_err_list)
            if new_err < best_err:
                best_fix = better_fit
                best_inliers = all_inliers
                best_err = new_err
        n_iterations += 1
    if best_fix is None:
        raise ValueError("本次迭代未找到最优解")
    if return_all:
        return best_fix, best_inliers
    else:
        return best_fix


# 绘图
def plot(all_data, x_exact, y_exact, ransac_fit, ransac_data):
    x_noisy = all_data[:, 0]
    y_noisy = all_data[:, 1]
    linear_fit = np.linalg.lstsq(all_data[:, [0]], all_data[:, [1]], rcond=-1)[0]
    pylab.plot(x_noisy, y_noisy, "k.", label="data")
    pylab.plot(ransac_data[:, 0], ransac_data[:, 1], "bx", label="RANSAC data")
    pylab.plot(x_exact[:, 0], np.dot(x_exact, ransac_fit)[:, 0], "r", label="RANSAC fit")
    pylab.plot(x_exact[:, 0], y_exact[:, 0], label="exact_system")
    pylab.plot(x_exact[:, 0], np.dot(x_exact, linear_fit), label="linear fit")
    pylab.legend()
    pylab.show()


if __name__ == '__main__':
    # 生成数据
    all_data, x_exact, y_exact = create_test_data()
    # 实例化
    llsm = LinearLeastSquareModel()
    # ransac迭代,寻找最优解
    ransac_fit, ransac_data = ransac(all_data, llsm, 50, 1000, 7e3, 300, return_all=True)
    # 绘制最终图像
    plot(all_data, x_exact, y_exact, ransac_fit, ransac_data)
