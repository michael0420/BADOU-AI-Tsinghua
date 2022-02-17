"""
@author: 陈志海
@fcn: 透视变换
"""

import numpy as np


def getWarpMatrix(src_points, dst_points):
    assert (src_points.shape[0] == dst_points.shape[0]) and src_points.shape[0] >= 4

    num = src_points.shape[0]
    a = np.zeros((num*2, 8))
    for i in range(num):
        src = src_points[i, :]
        dst = dst_points[i, :]
        a[2*i, :] = [src[0], src[1], 1, 0, 0, 0, -src[0] * dst[0], -src[1] * dst[0]]
        a[2*i+1, :] = [0, 0, 0, src[0], src[1], 1, -src[0] * dst[1], -src[1] * dst[1]]

    a_matrix = np.mat(a)
    b_matrix = dst_points.reshape((-1, 1))
    b_matrix = np.matrix(b_matrix)
    warpMatrix = a_matrix.I * b_matrix
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


# main
src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
src = np.array(src)
dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
dst = np.array(dst)

warpMatrix = getWarpMatrix(src, dst)
print(warpMatrix)

