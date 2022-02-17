##=============================================================================#
import numpy as np
import random



#1、实现从数据集中随机选取初始的几个点
def randomPartion(n, n_data):
    idx = np.arange(n_data.shape[0])
    np.random.shuffle(idx)
    idx1 = idx[:n]
    idx2 = idx[n:]
    return idx1, idx2


#2、将初始选择的点根据最小二乘法获取最适合的模型
def model(n, xdata, ydata):#n是随机选取的样本个数；xdata是样本的横坐标，ydata是样本点的纵坐标
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
     #n 你需要根据的数据量进行修改
     # 循环累加
    for i in range(n):
        s1 = s1 + xdata[i] * ydata[i]  # X*Y，求和
        s2 = s2 + xdata[i]  # X的和
        s3 = s3 + ydata[i]  # Y的和
        s4 = s4 + xdata[i] * xdata[i]  # X**2，求和
        # 计算斜率和截距
        k = (s2 * s3 - n * s1) / (s2 * s2 - s4 * n)
        b = (s3 - k * s2) / n
    print(k,b)
    return k, b


#3、计算样本集中属于该模型的所有内点。
def getError(xdata, ydata, k, b):#k,b为随机选取样本根据最小二乘法获取的模型参数；
    y_fit = k.item() * xdata + b.item()
    #print(y_fit.shape)
    return abs(y_fit)



if __name__ == "__main__":
#4、创建初始样本集，生成理想数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20*np.random.random((n_samples, n_inputs))
    perfect_fit = 60*np.random.normal(size=(n_inputs,n_outputs))
    perfect_b = 1
    B_exact = A_exact*perfect_fit + perfect_b

    #加噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    #局外点
    if 1:
        out_points_num = 100
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        out_idx = idx[:out_points_num]
        A_noisy[out_idx] = 20 * np.random.random((out_points_num, n_outputs))
        B_noisy[out_idx] = 50 * np.random.normal(size= (out_points_num, n_outputs))
    all_data = np.hstack((A_noisy, B_noisy))
    n=10
    t = 500
    iterations = 0
    k=10
    #bestfit = None
    bestErr = np.inf #设置默认值
    best_inlier_idxs = None
    while iterations < k:
        in_idx, out_idx = randomPartion(n, all_data)
        in_points = all_data[:n]
        out_points = all_data[n:]
        k, b = model(n, in_points[:,0], in_points[:, 1])
        error= getError(out_points[:,0], out_points[:, 1], k, b)
        print(error)
        print('test_err = ', error <t)
        also_idxs = out_idx[error < t]
        print ('also_idxs = ', also_idxs)
        also_inliers = all_data[also_idxs,:]
        print(also_inliers)
        print(in_points.shape, also_inliers.shape)
        d = 150
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (in_points, also_inliers) )
            better_k, better_b = model(len(betterdata), betterdata[:,0], betterdata[:, 0])
            print(better_k, better_b)
            betterErr = getError(betterdata[:,0], betterdata[:,1], better_k, better_b)
            thisErr = np.mean(betterErr)
            if thisErr < bestErr:
                bestErr = thisErr
                best_inlier_idxs = np.concatenate((in_idx, also_idxs))

        iterations += 1


