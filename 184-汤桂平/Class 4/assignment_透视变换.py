# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/3/21


import numpy as np

    # 根据原始图像和目标图像坐标，求透视变换矩阵
def perspective_Transform(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    ''' 已知原始图像4个坐标（以x,y表示）和目标图像的4个坐标（以X'和Y‘表示），
        待入公式a11 * x + a12 * y + a13 - a31 * x * X' - a32 * X' * y = X'
              a21 * x + a22 * y + a23 - a31 * x * Y' - a32 * y * Y' = Y'
        以矩阵形式表示为dot(A, warpMatrix) = B，先把A和B以矩阵形式表示出来，然后通过矩阵求逆的方法求出warpMatrix
        warpMatrix = [a11, a12, a13, a21, a22, a23, a31, a32, a33]
    '''
    dim = 2 * src.shape[0]
    A = np.zeros([dim, dim])

    A[0] = [src[0, 0], src[0, 1], 1, 0, 0, 0, -src[0, 0] * dst[0, 0], -src[0, 1] * dst[0, 0]]
    A[1] = [0, 0, 0, src[0, 0], src[0, 1], 1, -src[0, 0] * dst[0, 1], -src[0, 1] * dst[0, 1]]
    A[2] = [src[1, 0], src[1, 1], 1, 0, 0, 0, -src[1, 0] * dst[1, 0], -src[1, 1] * dst[1, 0]]
    A[3] = [0, 0, 0, src[1, 0], src[1, 1], 1, -src[1, 0] * dst[1, 1], -src[1, 1] * dst[1, 1]]
    A[4] = [src[2, 0], src[2, 1], 1, 0, 0, 0, -src[2, 0] * dst[2, 0], -src[2, 1] * dst[2, 0]]
    A[5] = [0, 0, 0, src[2, 0], src[2, 1], 1, -src[2, 0] * dst[2, 1], -src[2, 1] * dst[2, 1]]
    A[6] = [src[3, 0], src[3, 1], 1, 0, 0, 0, -src[3, 0] * dst[3, 0], -src[3, 1] * dst[3, 0]]
    A[7] = [0, 0, 0, src[3, 0], src[3, 1], 1, -src[3, 0] * dst[3, 1], -src[3, 1] * dst[3, 1]]

    B = dst.reshape((8, 1))  # B = [X0', Y0', X1', Y1', X2', Y2', X3', Y3']

    A = np.mat(A)  # 为了求逆，必须把数组转换成矩阵
    warpMatrix = A.I * B  # 矩阵的求逆方法

    warpMatrix = np.array(warpMatrix).T[0]
    '''warpMatrix本是8*1矩阵，通过np.array方法转换成（8，1）的二维数组，然后通过.T方法装置成（1，8）的二维数组
       最后通过切片[0]最终得到(8,)的一维数组
    '''
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    # 给出4个原始图像坐标和4个目标图像坐标
    src = np.array([[3.0, 5.2], [10.2, 60.0], [5.3, -8.4], [20.4, 2.0]])
    dst = np.array([[20.5, 44.2], [35.2, 100.0], [60.8, 48.2], [88.8, 206.6]])
    warpMatrix = perspective_Transform(src, dst)
    print('透视变换矩阵：\n', warpMatrix)









