# 最小二乘法详细实现
import numpy as np


def cal(d_x, d_y):
    """
    :param d_x: 源数据x坐标
    :param d_y: 源数据y坐标
    :return: k 和 b
    """
    # 参数过滤
    if not len(d_x) == len(d_y):
        print("err")
        return 0, 0
    x1 = np.array(d_x)
    y1 = np.array(d_y)
    n = len(d_x)
    xy = np.sum(x1 * y1)
    x = np.sum(x1)
    y = np.sum(y1)
    x_square = np.sum(np.square(x1))
    k = (n * xy - x * y) / (n * x_square - x * x)
    b = (y / n) - k * x / n
    return k, b


xx = [1, 2, 3, 4]
yy = [6, 5, 7, 10]

xx = [1.1, 2, 2, 4.5, 7, 9, 19]
yy = [2.1, 2.4, 2.6, 2.8, 4, 1.2, 6]
k, b = cal(xx, yy)
print("k,b:", k, b)
