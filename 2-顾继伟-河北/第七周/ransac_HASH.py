# -*- coding:UTF-8 -*-

import numpy as np
import scipy as sp
import scipy.linalg as sl
import cv2


def ransac(data, model, n, k, t, d, return_all=False):
    iterations = 0  # 迭代次数
    bestmodel = None  # 模型中最合适的一个
    bestmodel_error = np.inf  # 给这个模型的误差赋个初始值：无穷大

    while iterations < k:  # 如果迭代次数小于设置的k值，就一直执行循环
        all_indexs1, all_indexs2 = data_split(n, data.shape[0])
        selected_data = data[all_indexs1]
        remained_data = data[all_indexs2]

        maybemodel = model.fit(selected_data)
        maybemodel_error = model.get_error(remained_data, maybemodel)
        same_also_indexs = all_indexs2[maybemodel_error < t]
        same_also_data = data[same_also_indexs, :]

        if (len(same_also_data) > d):
            betterdata = np.concatenate((selected_data, same_also_data))  # 将视为同类的数组拼接起来
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < bestmodel_error:
                bestmodel = bettermodel
                bestmodel_error = thiserr
                best_inlier_idxs = np.concatenate((all_indexs1, all_indexs2))  # 更新局内点,将新点加入
        iterations += 1

    if bestmodel is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestmodel, {'inliers': best_inlier_idxs}
    else:
        return bestmodel


class Linear_LeastSquareMethod:  # 线性最小二乘法
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 在垂直方向上堆叠数组
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, residues, rank, s = sp.linalg.lstsq(A, B)  # x: 最小二乘法计算出的数组（相当于模型）; residues:残差和；最小二乘法尝试拟合A和B，计算得到A与B之间的函数
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, model)  # 矩阵乘法，理解为将A带入临时模型（y=ax+b）得到新的值B_fit
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 计算残差平方和
        return err_per_point


def data_split(n, data):  # 数据分割
    all_indexs = np.arange(data)  # 计算data的数组下标
    np.random.shuffle(all_indexs)  # 打乱数组下标
    all_indexs1 = all_indexs[:n]
    all_indexs2 = all_indexs[n:]

    return all_indexs1, all_indexs2


############################################################################################################################

def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi

    ############################################################################################################################

    all_data = np.hstack((A_noisy, B_noisy))  # 需要测试的数据，(500,2)
    input_columns = range(n_inputs)  # range()生成从0开始，以n_inputs作终点的一维数组
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 列表解析，效率更高，1+0的数组什么东东？
    model = Linear_LeastSquareMethod(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resides, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, return_all=True)

    ############################################################################################################################

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],  # 经过迭代的模型
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],  # 迭代开始之前的随机斜线
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],  # 最小二乘法生成，用于第一次迭代的模型
                   label='linear fit')
        pylab.legend()
        pylab.show()


class two_HASH:
    # 均值哈希算法
    def aHash(self, img):
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)  # 基于4x4像素邻域的3次插值法
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # s为像素和初值为0，hash_str为hash值初值为''
        s = 0
        s1 = 0
        hash_str = ''
        # 遍历累加求像素和
        for i in range(8):
            for j in range(8):
                print(gray[i, j])
                s1 += 1
                s = s + gray[i, j]
        print(s1)
        # 求平均灰度
        avg = s / 64
        # 灰度大于平均值为1相反为0生成图片的hash值
        for i in range(8):
            for j in range(8):
                if gray[i, j] > avg:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    # 差值算法
    def dHash(self, img):
        # 缩放8*9
        img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
        # 转换灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hash_str = ''
        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(8):
            for j in range(8):
                if gray[i, j] > gray[i, j + 1]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    # Hash值对比
    def cmpHash(self, hash1, hash2):
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(hash1) != len(hash2):
            return -1
        # 遍历判断
        for i in range(len(hash1)):
            # 不相等则n计数+1，n最终为相似度
            if hash1[i] != hash2[i]:
                n = n + 1
        return n


if __name__ == "__main__":
    # test()

    img1 = cv2.imread('lenna.png')
    img2 = cv2.imread('lenna_noise.png')
    hash1 = two_HASH().aHash(img1)
    hash2 = two_HASH().aHash(img2)
    print(hash1)
    print(hash2)
    n = two_HASH().cmpHash(hash1, hash2)
    print('均值哈希算法相似度：', n)

    hash1 = two_HASH().dHash(img1)
    hash2 = two_HASH().dHash(img2)
    print(hash1)
    print(hash2)
    n = two_HASH().cmpHash(hash1, hash2)
    print('差值哈希算法相似度：', n)
