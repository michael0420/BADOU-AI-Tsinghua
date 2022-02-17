#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : bxh
# @Time     : 2021/12/13 22:29
# @File     : 透视变换.py
# @Project  : 作业

import numpy as np
import cv2

def warpPerspectiveMatrix(srcImg,dstImg):
    if srcImg.shape[0] != dstImg.shape[0] or srcImg.shape[0] < 4:
        return
    nums = srcImg.shape[0]
    A = np.zeros((nums * 2 , 8))
    B = np.zeros((nums * 2 , 1))
    for i in range(0,nums):
        A_i = srcImg[i,:]
        B_i = dstImg[i, :]
        A[2*i,:]=[A_i[0],A_i[1],1,0,0,0,
		-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        A[2 * i+1, :] = [0, 0, 0,A_i[0], A_i[1], 1, 
		-A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i+1] = B_i[1]
    A = np.mat(A)
    warpMatrix = A.I*B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix,warpMatrix.shape[0],values=1.0,axis=0)
    warpMatrix=warpMatrix.reshape((3,3))
    return warpMatrix


if __name__ == '__main__':
    img = cv2.imread("F:\qingxie.png")
    src = np.float32([[246, 45], [297, 97], [10, 406], [212, 453]])
    dst = np.float32([[10, 17], [87, 59], [4, 419], [218, 448]])
    m = warpPerspectiveMatrix(src,dst)
    result = cv2.warpPerspective(img,m,(337, 488))
    cv2.imshow("img",img)
    cv2.imshow("result", result)
    cv2.waitKey(0)


