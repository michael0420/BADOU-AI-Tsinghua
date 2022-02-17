#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2021/12/26 16:57:04
@Author : LuZhanglin
'''

import numpy as np

def warp_matrix(src, dst):
    """根据变换前后的点求变换矩阵
    \math:$$
        a_{11} * x + a_{12} * y + a_{13} - a_{31} * x * X^' - a_{32} * X^' * y = X^'
        a_{21} * x + a_{22} * y + a_{23} - a_{31} * x * Y^' - a_{32} * y * Y^' = Y^'
        $$
    Parameters
    ----------
    src : np.ndarray
        shape[num_points, 2]
    dst : np.ndarray
        shape[num_points, 2]
    """
    # 至少已知4个点
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    # 原
    A = np.zeros((2*nums, 8))
    B = np.zeros((2 * nums, 1))
    """a33=1的情况下
    ww = [a11, a12, a13, a21, a22, a23, a31, a32]
    
    ax = [x,   y,   1,   0,   0,   0, -x*X', -X'y]
    ww.T * ax    

    bx = [0,   0,   0,   x,    y,   1, -x*Y', -y*Y']
    ww.T * bx
    """
    
    for i in range(nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                        -A_i[0] * B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, 
                        -A_i[0] * B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]

    A = np.mat(A)
    """
    A * warpMat= B
    ||
    A^{-1} * B = A^{-1} * A * warpMat = warpMat
    """
    warpMat = A.I * B

    # 插入设为1的a33
    warpMat = np.array(warpMat).T[0]
    warpMat = np.insert(warpMat, warpMat.shape[0], values=1.0, axis=0)
    return warpMat.reshape((3, 3))


if __name__ == "__main__":
    # 调用opencv接口做透视变换
    import cv2
    img = cv2.imread("photo1.jpg")

    result = img.copy()
    print(img.shape)

    # 图像的点坐标
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    m = cv2.getPerspectiveTransform(src, dst)
    print("warpMat by opencv:")
    print(m)

    m2 = warp_matrix(src, dst)
    print("warpMat by hand:")
    print(m2)

    result2 = cv2.warpPerspective(result, m, (337, 448))
    cv2.imshow("src", img)
    cv2.imshow("result", result2)
    cv2.waitKey(0)

