"""
几种常用的数据标准化方法
"""
import numpy as np
import matplotlib.pyplot as plt


# 归一化的两种方法
def normalization_1(x):
    """
    数据的归一化处理，方法1，将数据统一映射到[0,1]区间上
    xi_new = (xi - x_min) / (x_max - x_min)
    """
    x_min = min(x)
    x_max = max(x)
    return [(float(i) - x_min) / float(x_max - x_min) for i in x]


def normalization_2(x):
    """
    数据的归一化处理，方法2，将数据统一映射到[-1,1]区间上
    xi_new = (xi - x_mean) / (x_max - x_min)
    """
    x_mean = np.mean(x)
    x_min = min(x)
    x_max = max(x)
    return [(float(i) - x_mean) / float(x_max - x_min) for i in x]


def z_score(x):
    """
    z-score标准化（零均值归一化zero-mean normalization）：
    • 经过处理后的数据均值为0，标准差为1（正态分布）
    • 其中μ是样本的均值， σ是样本的标准差
    xi_new = (xi − μ) / σ
    """
    u = np.mean(x)
    sigma = np.std(x)
    return [(float(i) - u) / sigma for i in x]


if __name__ == '__main__':
    data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
            11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    # 统计data数据中每个元素出现的次数
    counts = []
    for d in data:
        num = data.count(d)
        counts.append(num)
    print('data = ', data)
    print('counts = ', counts)
    # 采用三种方法对data数据进行标准化
    data_normal1 = normalization_1(data)
    data_normal2 = normalization_2(data)
    data_z_score = z_score(data)
    # 绘图
    plt.plot(data, counts, color='m', linestyle=':', label='Raw data')
    plt.plot(data_normal1, counts, color='r', linestyle='--', label='normalization_1')
    plt.plot(data_normal2, counts, color='g', linestyle='-.', label='normalization_2')
    plt.plot(data_z_score, counts, color='b', label='z_score')
    plt.legend()
    plt.show()
