import numpy as np
import matplotlib.pyplot as plt

"""
使用RANSAC思想寻找一组数据点(二维)的线性方程y=kx+b的相对最优参数k,b；
每次迭代使用最小二乘法拟合k,b
"""


# 1. 编写最小二乘法算法
class LeastSquaresMethod:
    """
    最小二乘法：可以拟合y = k x + b，返回k,b还有残差平方和。
    """

    def __init__(self, yx):
        """
        接收数组yx,其中yx.shape=(2,n)
        """
        self.y = yx[0, :]
        self.x = yx[1, :]
        self.k = 1
        self.b = 0
        self.ssr = 0  # ssr = Sum of Squared Residuals

    def fit_model(self):
        """
        利用公式求解模型参数
        公式数学推导参考自己的blog: https://www.cnblogs.com/WarnerJDoe/p/16197021.html
        """

        numerator = self.y.shape[0] * np.sum(self.y * self.x) - np.sum(self.x) * np.sum(self.y)
        denominator = self.y.shape[0] * np.sum(self.x ** 2) - np.sum(self.x) ** 2
        self.k = numerator / denominator
        self.b = np.sum(self.y - self.x * self.k) / self.y.shape[0]

    def get_sum_squared_error(self):
        """求得残差平方和"""

        self.ssr = np.sum((self.y - self.x * self.k - self.b) ** 2)


# 2. 需要定义一个数据集生成器，生成(xi,yi)的集合供RANSAC使用
def get_samples(k, b, n, rand_riato=0.12):
    """
    生成样本点集合
    :param k: 真实斜率
    :param b: 真实截距
    :param n: 样本点个数
    :param rand_riato: 为占rand_riato比例的样本部分数据点添加随机噪声使其作为真实离群点
    :return: 包含inliers和outliers的数据集(xi,yi)
    """

    x_arr = 20 * np.random.random(size=(n, 1))  # 生成初始n个样本点的x值，x取值范围0~20
    samples_arr = x_arr * [k, 1]  # y = kx
    samples_arr[:, 0] += b  # y = kx + b
    samples_arr = samples_arr.T
    exact_arr = samples_arr.copy()
    # 接下来开始加噪声，模拟真实情况带来的误差和异常值
    samples_arr += np.random.normal(size=(2, n))
    samples_arr[:, :int(n * rand_riato)] += 30 * np.random.random(size=(2, int(n * rand_riato)))
    return samples_arr, exact_arr


# 3. RANSAC
def ransac_lsm(sample, res_max, n, n_min=0.6, times_max=9000):
    """
    :param sample: 样本的y值和x值
    :param res_max: 设定判断内群点(inliers)的残差最大值
    :param n: 每次迭代初选内群(inliers)数量
    :param n_min: 模型拥有全部内群点(inliers)最小数量
    :param times_max: 最大迭代次数
    :return: 得到的一个最好的内群集合arr_inliers
    """

    t = 0  # 记录迭代次数
    n_inliers = 0  # 记录每轮迭代结束时内群点个数
    while not (t > times_max or n_inliers >= int(sample.shape[1] * n_min)):
        index_random = np.arange(sample.shape[1])  # 生成与样本点数相同的索引序列
        np.random.shuffle(index_random)  # 打乱索引序列
        # print(f"随机选取索引值为：\n{index_random}")
        solution_lsm = LeastSquaresMethod(sample[:, index_random[:n]])  # 随机取n个点作为内群点
        solution_lsm.fit_model()  # 使用最小二乘法求解，返回k和b
        arr_res = sample[0, index_random[n:]] - (solution_lsm.k * sample[1, index_random[n:]] + solution_lsm.b)
        arr_res_index = np.argwhere(arr_res ** 2 <= res_max ** 2).flatten()  # 取得所有剩余点中符合内群点的索引
        if arr_res_index.shape[0] + n > n_inliers:  # 如果新得到的内群数量 > 记录的最大的内群数量
            # 存储所有内群点并更新记录最大内群数量
            arr_inliers = np.append(sample[:, index_random[n:]][:, arr_res_index], sample[:, index_random[:n]], axis=1)
            n_inliers = arr_inliers.shape[1]
        t += 1
    best_solution = LeastSquaresMethod(arr_inliers)
    best_solution.fit_model()
    best_solution.get_sum_squared_error()
    print(f'经{t}轮RANSAC得到包含{n_inliers}个内群点的模型：y = {best_solution.k} * x + {best_solution.b}')
    return best_solution.k, best_solution.b, best_solution.ssr, arr_inliers


if __name__ == "__main__":
    # 1. 得到随机点集合与设定的真实直线上的部分点集合
    temp_arr, real_arr = get_samples(2, 1, 500)
    # 2. RANSAC迭代得到k,b,ssr(残差平方和)和对应内群点集合
    best_k, best_b, best_ssr, best_inliers = ransac_lsm(temp_arr, 1.6, 15, times_max=10000)
    # 3. 直接使用最小二乘法拟合所有随机点集合的结果
    absolute_lsm = LeastSquaresMethod(temp_arr)
    absolute_lsm.fit_model()
    absolute_lsm.get_sum_squared_error()
    # 4. 只用添加高斯噪声的点集所拟合的结果
    real_lsm = LeastSquaresMethod(temp_arr[:, 160:])
    real_lsm.fit_model()
    real_lsm.get_sum_squared_error()

    # 5. 画图
    plt.plot(temp_arr[1, :], temp_arr[0, :], 'k.', label='data')
    plt.plot(best_inliers[1, :], best_inliers[0, :], 'rx', label='RANSAC data')
    plt.plot(real_arr[1, :], real_arr[0, :], label='exact system')

    best_inliers_y = best_inliers[1, :].reshape((best_inliers.shape[1], 1)) * [best_k, 1]  # y = kx
    best_inliers_y[:, 0] += best_b  # y = kx + b
    best_inliers_y = best_inliers_y.T
    plt.plot(best_inliers[1, :], best_inliers_y[0, :], label='RANSAC system')

    absolute_lsm_y = temp_arr[1, :].reshape((temp_arr.shape[1], 1)) * [absolute_lsm.k, 1]  # y = kx
    absolute_lsm_y[:, 0] += absolute_lsm.b  # y = kx + b
    absolute_lsm_y = absolute_lsm_y.T
    plt.plot(temp_arr[1, :], absolute_lsm_y[0, :], label='lsm system')

    plt.legend()
    plt.show()

