#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
'''
A为源点坐标
B为目标点坐标
'''
def main(A,B):
    matix = np.zeros([8,8])
    exp = np.zeros([8, 1])
#   验证A、B的有效性，暂时忽略
    for i in range(4):
        matix[2*i, 0] = A[i][0] #X0
        matix[2*i, 1] = A[i][1] #Y0
        matix[2*i, 2] = 1
        matix[2*i, 6] = -A[i][0]*B[i][0] #-X0*X'0
        matix[2*i, 7] = -A[i][1]*B[i][0] #-Y0*X'0
        exp[2*i] = B[i][0]
        matix[2*i+1, 3] = A[i][0]
        matix[2*i+1, 4] = A[i][1]
        matix[2*i+1, 5] = 1
        matix[2*i+1, 6] = -A[i][0] * B[i][1]  # -X0*Y'0
        matix[2*i+1, 7] = -A[i][1] * B[i][1]  # -Y0*Y'0
        exp[2 * i +1] = B[i][1]

    matix = np.mat(matix)
    wrapMatrix = matix.I * exp
    wrapMatrix = np.array(wrapMatrix)
    wrapMatrix = np.append(wrapMatrix,1)
    wrapMatrix = np.reshape(wrapMatrix,[3,3])

    return wrapMatrix

if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    wrapMatrix = main(src,dst)
    print(wrapMatrix)