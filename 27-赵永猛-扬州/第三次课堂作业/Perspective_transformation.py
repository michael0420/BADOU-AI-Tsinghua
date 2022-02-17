import numpy as np


def getWrapMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(nums):
        A[2*i, :] = [src[i][0], src[i][1], 1, 0, 0, 0, -src[i][0]*dst[i][0], -src[i][1]*dst[i][0]]
        A[2*i+1, :] = [0, 0, 0, src[i][0], src[i][1], 1, -src[i][0]*dst[i][1], -src[i][1]*dst[i][1]]
        B[2 * i] = dst[i][0]
        B[2 * i + 1] = dst[i][1]
    print(A, B)
    A = np.mat(A)
    w_matrix = A.I*B #用A.I求出A的逆矩阵
    w_matrix = np.array(w_matrix).T[0]
    w_matrix = np.insert(w_matrix, w_matrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    w_matrix = w_matrix.reshape((3, 3))
    return w_matrix


if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    src = np.array(src)
    dst = np.array(dst)
    a = getWrapMatrix(src, dst)
    print(a)

