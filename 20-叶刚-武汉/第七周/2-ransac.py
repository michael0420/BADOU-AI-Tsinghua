"""
RANSAC优化算法示例，以最小二乘法线性模型作为输入模型
"""

import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


class LinearLeastSquareModel:
    # 最小二乘求线性解, 后面将用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        # A = np.vstack([data[:, i] for i in self.input_columns]).T
        # B = np.vstack([data[:, i] for i in self.output_columns]).T
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        # 调用scipy框架的lstsq()函数求方程Ax=B的最小二乘解params（即求参数a，b组成的向量）
        # lstsq()函数，返回值依次为：params：最小二乘解向量；residues：每列的残差和；rank：a的有效秩；s：a的奇异值
        params, residues, rank, s = lstsq(A, B)
        # 返回最小二乘解向量，由解向量即可得到拟合的模型 y = params[0] + params[1]*x
        return params

    def get_error(self, data, model):
        # A = np.vstack([data[:, i] for i in self.input_columns]).T
        # B = np.vstack([data[:, i] for i in self.output_columns]).T
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        # 对新的数据矩阵A，B，求模型的拟合解 B_fit = model[0] + model[1]*A
        B_fit = np.dot(A, model)
        # axis=1，对每一行的元素求残差的平方和
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def data_random_partition(num_data, n):
    """
    作用：用于RANSAC中随机选择 n 个点作为内群
    随机分割数据，返回数据的两个部分，一部分为n行数据，另一部分为剩余的数据
    输入：
        num_data：data的行数
        n：选取n行数据
    返回：
        return n random rows of data and the other len(data)-n rows
    """
    all_idxs = np.arange(num_data)  # 获取src_data下标索引
    if len(all_idxs) < n:
        raise ValueError("len(num_data) < n")
    np.random.shuffle(all_idxs)  # 将下标索引打乱
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


def ransac(data, model, min_num_point, max_iters, threshold, d, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定（如线性最小二乘模型）
        min_num_point - 生成模型所需的最少样本点个数
        max_iters - 最大迭代次数
        threshold - 阈值：作为判断点是否属于内群的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
        debug - 是否打印中间结果
        return_all - 控制返回值的个数
    输出:
        bestfit - 最优拟合解（如果未找到，返回nil（无穷大））
    """
    # 0.初始化参数
    iterations = 0
    best_model = None
    best_err = np.inf  # 设置默认值，无穷大
    best_innergrop_idxs = None
    best_iteration = 0
    while iterations < max_iters:
        # 1.随机选择一些点作为内群
        select_idxs, other_idxs = data_random_partition(data.shape[0], min_num_point)
        select_points = data[select_idxs, :]
        other_points = data[other_idxs, :]
        # 2.由内群，计算适合内群的模型
        current_fit_model = model.fit(select_points)
        # 3.把未选中的点带入刚才的模型，计算每个点的误差，由误差判断这些点之中哪些点可以归为内群
        other_points_errors = model.get_error(other_points, current_fit_model)
        also_idxs = other_idxs[other_points_errors < threshold]
        also_points = data[also_idxs, :]
        if debug:
            print('====================================================================================')
            print('iteration:{}, len(also_points)={}'.format(iterations, len(also_points)))
            print('other_points_errors, mean={}, min={}, max={}'.format(np.mean(other_points_errors),
                                                                        other_points_errors.min(),
                                                                        other_points_errors.max()))
        # 4.当模型有效时，使用所有的内群点重新拟合模型
        # 判断当前模型是否有效（若内群点数量大于模型需要的最小点的数量，即给定值d，则模型有效）
        if len(also_points) > d:
            # 将所有内群点连接起来
            better_data = np.concatenate((select_points, also_points))
            # 使用所有的内群点重新拟合模型
            better_model = model.fit(better_data)
            better_errors = model.get_error(better_data, better_model)
            # 平均误差作为新的误差
            this_err = np.mean(better_errors)
            # 5.若重新拟合的模型误差最小，则将该模型作为最佳模型
            # 与PPT上的标准不同，PPT上是取内群点数量最多的那次建立的模型为最佳模型
            if this_err < best_err:
                best_model = better_model
                best_err = this_err
                best_innergrop_idxs = np.concatenate((select_idxs, also_idxs))
                best_iteration = iterations
        iterations += 1
    if best_model is None:
        raise ValueError("Didn't meet fit acceptance criteria!")
    if return_all:
        print("Best result: iter={}, error={}".format(best_iteration, best_err))
        return best_model, {'inliers': best_innergrop_idxs}
    else:
        print("Best result: iter={}, error={}".format(best_iteration, best_err))
        return best_model


def run_demo(add_outliers=True):
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据, shape=(500, 1)

    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率, shape=(1, 1)
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k，按照上一步的斜率，计算Y值

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1列向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1列向量,代表Yi

    # 添加“离群点”，将n_samples个样本中的n_outliers个样本变为“离群点”
    if add_outliers:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 取100个
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 将A_noisy中的部分样本变为离群点（0~20之间的数字）
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 将B_noisy中的部分样本变为离群点

    # 创建最小二乘线性模型
    all_data = np.hstack((A_noisy, B_noisy))  # 水平堆叠，形式([Xi,Yi]....) shape:(500,2)
    input_columns = range(n_inputs)  # 输入X由all_data的[0: n_inputs]列组成
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 输出Y由all_data的[n_inputs：n_inputs+n_outputs]列组成
    model = LinearLeastSquareModel(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    # 运行RANSAC算法，得到最优模型和内群点
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 250, debug=True, return_all=True)

    # 调用scipy框架，求解最小二乘线性解（作为结果比较）
    linear_fit, residues, rank, s = lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # 绘图，展示拟合结果
    # 绘制散点图、内群点
    """
    marker=['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_','.',',']
    color=['b','g','r','c','m','y','k','w']
    linestyle=['-','--','-.',':']
    """
    plt.scatter(A_noisy[:, 0], B_noisy[:, 0], marker='x', s=15, color='m', label='data')
    plt.scatter(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], marker='p', s=30,
                color='y', alpha=0.5, label='RANSAC data')
    # 绘制RANSAC拟合结果
    plt.plot(A_exact[:, 0], np.dot(A_exact, ransac_fit)[:, 0], color='r', label='RANSAC fit')
    # 绘制最小二乘拟合结果
    plt.plot(A_exact[:, 0], np.dot(A_exact, linear_fit)[:, 0], color='g', label='linear fit')
    # 绘制精确结果
    plt.plot(A_exact[:, 0], np.dot(A_exact, perfect_fit)[:, 0], color='b', label='exact system')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_demo(add_outliers=True)
