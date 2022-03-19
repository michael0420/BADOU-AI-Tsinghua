#coding:utf-8

'''
# 算法：
输入：
data //一组观测数据
model //假设模型，事先自己确定
n //适应于模型的最少数据个数，如果是直线的话，n=2
k //算法的迭代次数
t - 阈值:作为判断点满足模型的条件
d - 拟合较好时,需要的样本点最少的个数,当做阈值看待/判定模型是否适用于数据集的数据个数，人为设定.
输出：
bestfit - 最优拟合解（返回nil,如果未找到）
'''

import cv2 as cv
import numpy as np
import scipy as sp
import scipy.linalg as sl

def ransac(data, model,n, k, t, d, debug=False, return_all=False):
    iterations = 0   #初始迭代次数
    bestfit = None  #最好的参数
    besterr = np.inf #设置默认值。表示+∞，是没有确切的数值的,类型为浮点型
    best_inlier_idxs = None #最好的局内点的索引
    while iterations < k:
        maybe_idxs, test_idxs = random_parition(n, data.shape[0]) #每次随机选取的局内点，局外点
        # print('test_indxs\n', test_idxs,test_idxs.shape)    # 多少个shape(450,)索引
        maybe_inliers = data[maybe_idxs, :]  #获取size(maybe_indxs)局内点每行数据（Xi,Yi）
        # print('maybe_inliers', maybe_inliers, maybe_inliers.shape)  # 多少个shape(50, 2)数据
        test_points = data[test_idxs]  #其他测试点数据（Xi,Yi）
        # print('test_points', test_points, test_points.shape)  # 多少个shape(450, 2)数据
        maybe_model = model.fit(maybe_inliers)  #拟合模型 # y=k*x
        # print('maybe_model', maybe_model, maybe_model.shape)    # 返回的是斜率K
        test_err = model.get_error(test_points, maybe_model)  #计算误差：平方和最小 shape（450,）
        # print('test_err', test_err, test_err.shape)   # 每个局外点点的误差 (450,)
        # print('test_err\n', test_err < t)       # 返回bool型的
        also_idxs = test_idxs[test_err < t]     # 未超过阈值的数据，局外点中也是局内点的索引
        # print ('also_idxs = \n', also_idxs, also_idxs.shape)    # shape(true的个数, )
        also_inliers = data[also_idxs,:]
        # print('also_inliers', also_inliers, also_inliers.shape) #也是局内点的点
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )
        # if len(also_inliers > d):
#         print('d = ', d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) ) #样本连接.#数组拼接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs) #平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) ) #更新局内点,将新点加入
        iterations += 1

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        print(bestfit, '->', best_inlier_idxs)
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit

#随机划分
def random_parition(n, all_data):
    all_indxs = np.arange(all_data)  #获取n_data下表索引,n_data=500
    np.random.shuffle(all_indxs)  #打乱所有下标索引
    indxs1 = all_indxs[:n]  # 前n个随机的索引，随机选取的局内点
    # print('indxs1:\n', indxs1, indxs1.shape)    #(50, 0)，随机选取局内点的索引
    indxs2 = all_indxs[n:]  # 随机选取的局外点的索引
    # print('indx2', indxs2, indxs2.shape)    #shape(450, ),索引
    return indxs1, indxs2   #

#定义线性最小二乘法的类 用最小二乘法求线性解，用于ransac的输入模型
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns      #1
        self.output_columns = output_columns    #1
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 数据第一列Xi -> 行Xi #此时input_columns=0
        # print('A', A, A.shape)      #（50x1）多少个，
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 数据第二列Yi -> 行Yi #此时output_columns=[1]
        # print('B', B, B.shape)      #（50x1）多少个，
        x, resids, rank, s = sl.lstsq(A, B)  # x[0]=a,x[1]=b, residues:残差和
        # print('x:\n', x, x.shape)   # shape(1,1)
        return x  # 返回最小平方和向量 （a, b）#注意此模型中没有偏置a,给予一个b

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T   #此时input_columns=0
        # print('A', A, A.shape)      # shape(450,1),X值
        B = np.vstack([data[:, i] for i in self.output_columns]).T  #此时output_columns=1
        # print('B:\n', B, B.shape)   # shape(450, 1),Y值
        B_fit = np.dot(A, model)  # 计算的y值，B_fit = model.K * A + model.b #此时没有偏置b,y=kx
        # print(B_fit)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row # 每一行进行相减再平方,返回的是（450,）
        # print('err_per_point', err_per_point, err_per_point.shape)  # shape(450,)
        # print('--------')
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量    ##浮点数范围 : (0,1),(500行，1列)
        # print('A_exact\n', A_exact)  # (500x1)
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率    ##输出一个二维形式的一个值如[[2,34]]
    #     print('perfect_fit\n', perfect_fit)  #(1x1)
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k。矩阵乘（500x1x1x1=500x1）
    #     print('B_exact\n', B_exact)  # y=(500x1)

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi
    #     print('A，B加噪声\n', A_noisy, A_noisy.shape, B_noisy)  # shape(500x1)

    if 1:  # Ture
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499 ##步长为1
        #         print('500个索引all_ibxs\n', all_idxs, all_idxs.shape)  # shape(500,)
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        #         print('100个随机局外点的索引：\n', outlier_idxs, outlier_idxs.shape)  #打乱的100个索引shape(100,)
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        #         print('A加入噪声,局外点：\n', A_noisy, A_noisy.shape)  ##(500x1)
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    #         print('B加入噪声,局外点：\n', B_noisy, B_noisy.shape)  ##(500x1)
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列 ##在水平方向上平铺
    #     print('all_data:\n', all_data, all_data.shape)  ##(500x2)
    input_columns = range(n_inputs)  # 数组的第一列x:0
    # print('input_columns:', input_columns)  ##(0,1)。可以和下面的可以互换
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    # print('output_columns:', output_columns)  ##[1]
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])  ##对所有数据做最小二乘法

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])       #返回的是元素值从小到大排序后的索引值的数组，升序。#
        # print('sort_idxs', sort_idxs, sort_idxs.shape)  #(500, )
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组
        # print('A_col0_sorted', A_col0_sorted, A_col0_sorted.shape)  #(500,1)

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图 #黑点
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")     # x号
            # print('和ransac_data一样:',ransac_data['inliers'])
        # else:
        #     pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
        #     pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')      #随机采样一致性拟合的直线
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')    #随机拟合的直线
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')      #最小二乘法拟合的直线
        pylab.legend()
        pylab.show()

if __name__ == "__main__":
    test()