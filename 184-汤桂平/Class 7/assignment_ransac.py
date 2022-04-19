# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/3/28

import numpy as np
import scipy as sp
import scipy.linalg as sl

'''
输入要选择的内群点数量和总的样本数，随机取n个内群点,返回选取的n_inlier个内群点和剩余点的索引，即行标
'''
def random_partition(n_inlier, n_data):
    all_index = np.arange(n_data)
    np.random.shuffle(all_index)
    idx1 = all_index[:n_inlier]
    idx2 = all_index[n_inlier:]
    return idx1, idx2

'''
定义类，利用最小二乘法把得到的n个内群点进行拟合。fit方法返回的第一个向量就是斜率和截距，其中x[0]是截距，x[1]是斜率
geterr()函数返回每个点的误差组成的向量
'''
class leastSquare:
    def __init__(self, input_columes, output_columes, debug=False):  # 类的初始化方法
        self.input_columes = input_columes
        self.output_columes = output_columes

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columes]).T  # 行Xi
        B = np.vstack([data[:, i] for i in self.output_columes]).T  # 行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # 调用最小二乘法函数得到x为拟和的模型，即斜率和截距组成的向量
        return x  # 返回最小平方和向量

    def geterr(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columes]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columes]).T
        B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 计算通过模型计算的Yi值与实际的Yi的误差
        return err_per_point   # 返回误差向量

'''
利用ransac算法得到最佳的拟合函数
'''
def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    '''
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解
    '''
    iteration = 0     # 迭代次数计数器
    bestfit = None    # 最佳拟合解存储变量
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iteration < k:   # 迭代次数须小于k
        maybe_idx, test_idx = random_partition(n, data.shape[0])    # 把全部样本分成用于拟合模型的数据和测试模型的数据
        maybe_data = data[maybe_idx, :]  # 用于拟合模型的数据
        test_data = data[test_idx, :]
        maybe_model = model.fit(maybe_data)  # 拟合选取的内群数据得到模型
        test_err = model.geterr(test_data, maybe_model)  # 计算模型与实际样本点的误差，返回误差向量
        print('test error=', test_err < t)   # 误差小于阈值t，则为True，大于t为False，组成bool型列表
        also_idx = test_idx[test_err < t]    # 返回的also_idx为对应索引为True的test_idx的元素形成的列表
        print('also_idx:', also_idx)
        also_inlier = data[also_idx, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinlier) = %d' % (iteration, len(also_inlier)))
        print('d:', d)
        if (len(also_inlier) > d):   # 在误差内的点数量大于d才会被认可
            betterdata = np.concatenate((maybe_data, also_inlier))
            bettermodel = model.fit(betterdata)
            bettererr = model.geterr(betterdata, bettermodel)
            thiserr = np.mean(bettererr)
            if thiserr < besterr:
                besterr = thiserr
                bestfit = bettermodel
                best_inlier_idxs = np.concatenate((maybe_idx, also_idx))
        iteration += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit

def test():
    '''
    step1:获取用于拟合的数据：先通过随机方法np.random.random得到无噪声的x，再随机生成一个斜率k，然后利用y=kx+b得到y
    再分别在x和y上添加随机高斯噪声。最后利用np.random.shuffle方法随机选定要添加离群点的坐标，并用np.random.normal方法
    得到离群点(x,y)替换掉原始数据，从而得到最终的用于拟合的数据all_data
    '''
    n_sample = 500  # 总的样本点数量
    n_input = 1     # 输入的自变量数量
    n_output = 1    # 输出的变量数量
    A_exact = 20 * np.random.random((n_sample, n_input))    # 随机生成500个x
    k_array = 50 * np.random.normal(size=(n_input, n_output))  # 随机生成斜率k，为了适应多个输入、输出的情况把k写成向量形式
    B_exact = np.dot(A_exact, k_array)  # 利用y=k*x得到y
    # 添加高斯噪声，避免所有点都在一条直线上，且高斯噪声用最小二乘法可以很好地处理
    A_noise = A_exact + np.random.normal(size=A_exact.shape)   # 添加随机高斯噪声给x
    B_noise = B_exact + np.random.normal(size=B_exact.shape)   # 添加随机高斯噪声给y
    # 添加离群点：通过随机方式选定要添加离群点的位置，并且在要添加离群点的地方添加高斯噪声
    n_out = 100   # 设置离群点的数量
    all_index = np.arange(A_noise.shape[0])    # 所有数据的索引
    np.random.shuffle(all_index)               # 打乱索引，从而随机选择要添加离群点的坐标
    out_index = all_index[:n_out]
    A_noise[out_index] = 20 * np.random.normal(size=(n_out, n_input))
    B_noise[out_index] = 20 * np.random.normal(size=(n_out, n_input))
    all_data = np.hstack((A_noise, B_noise))  # 得到最终的500个样本点，shape为（500，2）其中第0列为x的值，第1列为对应y的值
    input_columes = range(n_input)  # x对应的列
    output_columes = [n_input + i for i in range(n_output)]   # y对应的列


    debug = False
    '''
    step2: 实例化最小二乘法类
    '''
    model = leastSquare(input_columes, output_columes, debug=False)   # 利用最小二乘法得到拟合模型
    '''
    step3: 调用ransac函数，得到最佳拟合函数和最佳拟合函数对应的拟合点的索引
    '''
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noise[:, 0], B_noise[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noise[ransac_data['inliers'], 0], B_noise[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noise[non_outlier_idxs, 0], B_noise[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noise[outlier_idxs, 0], B_noise[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()

if __name__ == "__main__":
    test()






