import numpy as np
import scipy as sp
import scipy.linalg as sl


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    #init方法 对象初始化时调用
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.array([data[:,i] for i in self.input_columns])
        B = np.array([data[:,i] for i in self.output_columns])
        A = np.vstack([A**0,A**1]).T  # 将输入数据的自变量数据组成新的数组，并变为列数据
        B = np.vstack(B).T            # 将输出数据的因变量数据组成新的数组，并变为列数据
        x, resids, rank, s = sl.lstsq(A, B)  # #求取各个系数大小 可求得f(x)=a+bx里的a和b。a和b记录在lstsq函数的第一个返回值里。
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.array([data[:, i] for i in self.input_columns])
        B = np.array([data[:, i] for i in self.output_columns])
        B = np.array([data[:, i] for i in self.output_columns])
        A = np.vstack([A ** 0, A ** 1]).T  # 将输入数据的自变量数据组成新的数组，并变为列数据
        B = np.vstack(B).T  # 将输出数据的因变量数据组成新的数组，并变为列数据
        B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum的目的是有部分模型输出不是单输出，需要将所有的平方和误差进行相加
        return err_per_point


def random_partition(n, n_data):
    '''
    函数用途：数据下标拆分，随机在数据n_data个数中随机拆分出选择点
    返回值：返回拆分后的下标列表
    '''
    all_idxs = np.arange(n_data) # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]         # 选取前n个下标索引
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    '''
    函数用途：实现ransac：随机采样一致性
    函数参数：
        data：样本点
        model：合适的内群模型
        n：生成模型所需要的最少的样本点
        k：最大的迭代次数
        t：阈值：作为判断模型参数好坏的条件
        d：拟合较好时，需要的样本点最少的个数，作为阈值看待
        debug = False：
        return_all = False：
    返回值：最优的拟合解 （返回为nil,为未找到最优解）
    '''
    iterations_num = 0 # 迭代次数
    bestfit = None    # 最好模型中的参数
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None   # 最好模型中的内群
    while iterations_num < k:
        # （设置内群） 随机选择 最小样本个点 作为 内群点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print ('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :] #获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs,:] #若干行(Xi,Yi)数据点

        # （计算合适内群模型） 用 内群 进行线性回归
        maybemodel = model.fit(maybe_inliers) #拟合模型
        print('可能权重：',maybemodel)

        # （将适合模型的外群点加入内群点中）
        test_err = model.get_error(test_points, maybemodel) #计算误差:所有点平方和误差
        print('test_err = ', test_err <t)
        also_idxs = test_idxs[test_err < t]   # 得到外群点中 满足模型阈值条件的点 的下标
        print ('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs,:]      # 将符合条件的外群点选出
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )
        # if len(also_inliers > d):
        print('d = ', d)



        # （添加内群，重新拟合模型，计算误差，添加部分点后，是否该模型比最好的模型好）
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) ) #样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs) #平均误差作为新的误差
            if thiserr < besterr:    # 得到较低的误差，更新数据
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) ) #更新局内点,将新点加入
        iterations_num += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit





if __name__ == "__main__":
    #生成 一次函数 数据
    n_samples = 500 # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    b_fit = 10 * np.random.random()  # 随即生成一个截矩b
    B_exact = np.dot(A_exact, perfect_fit) + b_fit  # y = x * k + b
#    print('原始设定：y=',b_fit,'+',perfect_fit,'*x')

    #加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal( size = A_exact.shape ) #500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal( size = B_exact.shape ) #500 * 1行向量,代表Yi


    # 添加"局外点" 个数：100个
    # 打乱方法： 将顺序点的索引打乱，根据索引值找出数据点，并将其替换为句外点
    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
    np.random.shuffle(all_idxs)  # 将all_idxs打乱
    outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
    A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
    B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi

    # 单线性回归
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    linear_fit = model.fit(all_data)    # b = linear_fit[0]  k = linear_fit[1]
#    print('线性回归：y=',linear_fit[0],'+',linear_fit[1],'*x')

    #run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True)

    print('原始设定：y=', b_fit, '+', perfect_fit, '*x')
    print('线性回归：y=', linear_fit[0], '+', linear_fit[1], '*x')
    print('ransac算法：y=', ransac_fit[0], '+', ransac_fit[1], '*x')



    import pylab
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组
    A_col0_sorted_t = A_col0_sorted.T
    a = np.vstack((A_col0_sorted_t ** 0, A_col0_sorted_t ** 1)).T

    if 1:
        pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
        pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    else:
        pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
        pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

    pylab.plot(A_col0_sorted[:, 0],
               np.dot(a, ransac_fit)[:, 0],
               label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(a,[[b_fit],[perfect_fit]])[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(a, linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()