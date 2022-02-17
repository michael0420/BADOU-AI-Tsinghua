#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import scipy.linalg as sl

"""
Ransac
1. 在数据中随机选择几个点设定为内群
2. 计算适合内群的模型 e.g. y=ax+b ->y=2x+3 y=4x+5
3. 把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群 e.g. hi=2xi+3->ri
4. 记下内群数量
5. 重复以上步骤
6. 比较哪次计算中内群数量最多,内群最多的那次所建的模型就是我们所要求的
"""

"""
准备测试数据
"""
def prepare_data():
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

    return A_noisy,B_noisy

"""最小二乘"""

def LeastSqaureMethod(data_x,data_y):
    if len(data_x)!=len(data_y):
        # 数据无效
        return None,None
    size_N = len(data_x)
    tmp_xy = 0
    tmp_x = 0
    tmp_y = 0
    tmp_x_square = 0
    for i in range(size_N):
        tmp_xy += data_x[i]*data_y[i]
        tmp_x += data_x[i]
        tmp_y += data_y[i]
        tmp_x_square +=data_x[i]*data_x[i]

    k = (size_N*tmp_xy-tmp_x*tmp_y)/(size_N*tmp_x_square-tmp_x*tmp_x)
    b = (tmp_y/size_N)-k*tmp_x/size_N
    return k,b

def ransac(A_noisy,B_noisy):
    size = len(A_noisy)
    final_k=0
    final_b=0
    #     在数据中随机选择几个点设定为内群，随机点个数为M个
    M = 20
    #最大循环次数
    max_cir = 30
    pre_neiqun = M

    for cir in range(max_cir):
        neiqun_A = np.zeros([M])
        neiqun_B = np.zeros([M])
        random_l = np.zeros([M])
        size_neiqun = M
        rands = np.random.randint(size-1,size = M)
        for i in range(M):
            random = rands[i]
            random_l[i] = random
            neiqun_A[i] = A_noisy[random]
            neiqun_B[i] = B_noisy[random]

        # 2. 计算适合内群的模型 e.g. y=ax+b ->y=2x+3 y=4x+5
        k,b = LeastSqaureMethod(neiqun_A,neiqun_B)
        #3. 把其它刚才没选到的点带入刚才建立的模型中，计算是否为内群 e.g. hi=2xi+3->ri
        threadhold = 100
        # 判断内群点的阈值
        for i in range(size):
            if i in random_l:
                # print("内群点")
                continue
            else:
                x = A_noisy[i]
                y_1 = B_noisy[i]
                y_2 = k*x+b
                #判断y_1和y_2距离是否小于阈值M
                if abs(y_2-y_1)<threadhold:
                    #满足条件，内群点
                    size_neiqun +=1
        if size_neiqun > pre_neiqun:
            final_k = k
            final_b = b
            pre_neiqun = size_neiqun
        if size_neiqun == size:
            return k,b

    return final_k,final_b

A_noisy,B_noisy = prepare_data()
k,b = ransac(A_noisy,B_noisy)
print(k,b)