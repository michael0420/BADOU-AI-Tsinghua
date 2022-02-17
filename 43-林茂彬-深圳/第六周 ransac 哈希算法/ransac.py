import numpy as np
import scipy as sp
import scipy.linalg as sl
import cv2
'''
RANSAC
算法的输入
1.一组观测数据（往往含有较大的噪声或无效点）-----数据
2.一个用于解释观测数据的参数化模型 -----确定的模型
3.一些可信的参数。------模型的参数
算法步骤
1.在数据中随机选择几个点设定为内群
2.根据所选择的内群数据计算适合内群的模型 y=ax+b
例如–>选择第一个内群数据时计算出y=2x+3 选择第二个内群数据时计算出y=4x+5…等
3.把其它刚才没选到的点带入刚才建立的模型中，计算模型输出值 hi=2xi+3 以及模型输出值与真实值的差值ri 判断该点书否为内群点 ri小于某个阈值则判断为内群点。
4.记下内群数量
5.重复以上步骤
6.比较哪次计算中内群数量最多 内群最多的那次所建的模型就是我们所要求的解
注意：不同问题对应的数学模型不同，因此在计算模型参数时方法必定不同 RANSAC 的作用不在于计
算模型参数，而是提供更好的输入数据（样本）。（这导致 ransac 的缺点在于要求数学模型已知）
原文链接：https://blog.csdn.net/m0_43609475/article/details/113994693
'''
'''
 输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
'''


def ransac(data, model, n, k, t, d, return_all=False):
    iterations = 0  # 迭代次数
    bestmodel = None  # 模型中最合适的一个
    bestmodel_error = np.inf  # 给这个模型的误差赋个初始值：无穷大

    while iterations < k:  # 如果迭代次数小于设置的k值，就一直执行循环   最大迭代次数
        all_indexs1, all_indexs2 = data_split(n, data.shape[0])  ## 生成模型所需的最少样本点 n 从样本中去N个随机样本点
        selected_data = data[all_indexs1]
        remained_data = data[all_indexs2]

        maybemodel = model.fit(selected_data) ##最小二乘法求出 拟合模型
        maybemodel_error = model.get_error(remained_data, maybemodel) ## 把其它刚才没选到的点带入刚才建立的模型中，计算模型输出值 hi=2xi+3 以及模型输出值与真实值的差值ri 判断该点书否为内群点 ri小于某个阈值则判断为内群点
        same_also_indexs = all_indexs2[maybemodel_error < t]
        same_also_data = data[same_also_indexs, :]  ##记下内群数量

        if (len(same_also_data) > d): ##比较哪次计算中内群数量最多 内群最多的那次所建的模型就是我们所要求的解
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

    all_data = np.hstack((A_noisy, B_noisy))  # 需要测试的数据，(500,2)   np.hstack():在水平方向上平铺
    input_columns = range(n_inputs)  # range()生成从0开始，以n_inputs作终点的一维数组
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 列表解析，效率更高，
    model = Linear_LeastSquareMethod(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resides, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])




    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, return_all=True)
    '''
     输入:
            data - 样本点
            model - 假设模型:事先自己确定
            n - 生成模型所需的最少样本点
            k - 最大迭代次数
            t - 阈值:作为判断点满足模型的条件
            d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
        输出:
            bestfit - 最优拟合解（返回nil,如果未找到）
    '''
    ############################################################################################################################

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

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