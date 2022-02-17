import numpy as np
import scipy
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
mpl.rcParams['font.sans-serif'] = 'KaiTi'

def ransac(data, model, n, k, t, d, debug=False, return_all=False):  # 定义ransac函数,
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
    """
    iterations = 0  # 迭代次数
    bestfit = None
    besterr = np.inf  # 设置默认值, np.inf 为正无穷
    best_inlier_idxs = None
    while iterations < k:  # 迭代次数未到达阈值
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  # data可能有多个维度
        # maybe_idxs 内群, test_idxs 外群
        maybe_inliers = data[maybe_idxs]  # 获取size(maybe_idxs)行数据(Xi,Yi,Zi...), 即内群点集
        test_points = data[test_idxs]  # 待定外群点集
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差,返回一个误差向量，对应每一个点的误差
        print('是否能归入内群:', test_err < t)
        also_idxs = test_idxs[test_err < t]  # 如果残差小于阈值, 则说明可用, 继承True的点集下标
        '''
         a=np.array(([1,2],[3,4],[5,6],[7,8]))
         b=np.array(([True False False True]))
         输出：a[b]=[[1,2],[7,8]]
        '''
        also_inliers = data[also_idxs]  # 通过下标索引将满足条件点集加入内群

        if len(also_inliers > d):
            print('d = ', d)
        if len(also_inliers) > d:   # 如果拟合较好,内群点大于阈值，则满足条件
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接，默认dim=0
            bettermodel = model.fit(betterdata)  # 对新的内群点进行拟合，得到向量
            better_errs = model.get_error(betterdata, bettermodel)  # 重新计算误差
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:  # 如果误差更小，则更新
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs)) # 返回下标
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria") # 报错
    if return_all:  # 是否返回所有点
        return bestfit,best_inlier_idxs
    else:
        return bestfit

def random_partition(n, n_data):
    all_idxs = np.arange(n_data)  # 创建与data大小相同的索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]  # 前n个索引样本点为内群点
    idxs2 = all_idxs[n:]  # 剩余的索引样本点为外群点
    return idxs1, idxs2


class LinearLeastSquareModel:

    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        z = np.vstack([data[:, i] for i in range(0, self.input_columns)]).T  # 第一列为Xi
        ch = np.array(([[1] for i in range(z.shape[0])]))  # 添加常数项
        A = np.hstack((ch, z))
        B = np.vstack([data[:, i] for i in range(self.input_columns, self.input_columns + self.output_columns)]).T  # 第二列为Yi
        x, residues, rank, s = scipy.linalg.lstsq(A, B)
        '''
        scipy.linalg.lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None)
        功能：输入左侧矩阵a,右侧矩阵b，求ax=b 中的x, x的大小取决于a与b
        输入：左侧矩阵a, 右侧矩阵b, 其他参数一般不用改
        返回：x 矩阵或向量，取决于a,b
             residues 残差和
             rank 矩阵a的秩
             s a的奇异值
        '''
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        z = np.vstack([data[:,i] for i in range(0,self.input_columns)]).T  # 第一列为Xi
        ch = np.array(([[1] for i in range(z.shape[0])]))  # 添加常数项
        A = np.hstack((ch,z))
        B = np.vstack([data[:,i] for i in range(self.input_columns,self.input_columns+self.output_columns)]).T  # 第二列为Yi
        B_fit = np.dot(A, model)  # 点积，得到一个向量，值为模型值
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def test():
    n_samples = 100  # 噪声样本个数
    n_inputs = 1  # 输入变量个数, 还有一个为常数项
    n_outputs = 1  # 输出变量个数
    data=30*np.random.random((n_samples,n_inputs+1))  # np.ramdom.random((50，2)) 生成50行2列的(0,1)浮点数
    plt.scatter([data[:, i] for i in range(0, 1)], [data[:, i] for i in range(1, 2)], c='b', marker='*')
    model = LinearLeastSquareModel(n_inputs, n_outputs, debug=False)
    X = np.linspace(0, 10, 400)
    Y = [3 * i + 10 + 2 * random.random() for i in X[:-20]] + [random.randint(0, int(i)) for i in X[-20:]]
    h = np.array(([[X[i], Y[i]] for i in range(len(Y))]))
    data=np.vstack((data,h))
    plt.scatter([data[:, i] for i in range(0, 1)], [data[:, i] for i in range(1, 2)], c='b', marker='*')
    k1,k2=ransac(data,model,30,1000,40,10,debug=False,return_all=True)
    plt.scatter([data[k2][:,i] for i in range(0,1)],[data[k2][:,i] for i in range(1,2)],c='r',marker='*')
    plt.title('RANSAC 算法一览')
    plt.show()
    print(k2)
    print(data)
test()