#coding:utf-8

import numpy as np

'''根据公式计算得到变换矩阵'''
# 求变换矩阵
def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4   ##assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。等价于 if not

    nums = src.shape[0]
    A = np.zeros((2*nums, 8))   # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]     #取第 i 行
        B_i = dst[i, :]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]   ##偶数行，公式相同

        B[2*i] = B_i[0]

        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]

        B[2*1+1] = B_i[1]

    A = np.mat(A)   ##数据被解释为矩阵
    print(A, '\n', B)
    #用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B    #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32  ## 8x1

    #结果后处理
    warpMatrix = np.array(warpMatrix).T[0]  ## 1x8, 取这一行，shape（8,）
    print('转置后的：\n', warpMatrix, warpMatrix.shape, '->', warpMatrix.shape[0])
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1   ##shape(9,)
    print(warpMatrix.shape[0])
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print('WarpMatrix: \n', warpMatrix)
