#!/usr/bin/env python 
# coding:utf-8
import cv2
import numpy as np


# 获取转换矩阵
def getWarpMatrix(src_matrix, dst_matrix):
    # 坐标的数量
    num = src_matrix.shape[0]
    # 由透视变换公式映射关系可得
    # 初始化矩阵A 8*8
    A_matrix = np.zeros((num * 2, 8))
    # 初始化矩阵B 8*1
    B_matrix = np.zeros((num * 2, 1))

    # 遍历对矩阵A和B进行赋值
    for i in range(num):
        # 原矩阵每行数据单独列出
        A_matrix_i = src_matrix[i, :]
        # 目标矩阵每行数据单独列出
        B_matrix_i = dst_matrix[i, :]
        # 对矩阵A进行赋值
        A_matrix[2 * i, :] = [A_matrix_i[0], A_matrix_i[1], 1, 0, 0, 0, -A_matrix_i[0] * B_matrix_i[0],
                              -A_matrix_i[1] * B_matrix_i[0]]
        # 对矩阵B进行赋值
        B_matrix[2 * i, :] = [B_matrix_i[0]]

        A_matrix[2 * i + 1, :] = [0, 0, 0, A_matrix_i[0], A_matrix_i[1], 1, -A_matrix_i[0] * B_matrix_i[1],
                                  -A_matrix_i[1] * B_matrix_i[1]]
        B_matrix[2 * i + 1, :] = [B_matrix_i[1]]

    # A*matrix = B
    # 获取A的逆矩阵
    inverse_matrix_A = (np.mat(A_matrix)).I
    # 此时为8*1 矩阵
    matrix = (inverse_matrix_A * B_matrix)
    # 转为数组
    matrix = np.array(matrix).T[0]
    # 将a33 =1 插入最后
    matrix = np.insert(matrix, matrix.shape[0], 1, axis=0)
    # 重置为3*3 矩阵
    final_warpMatrix = matrix.reshape((3, 3))
    return final_warpMatrix


# 将原图转为目标图
def warp_Perspective(src, M, dsize):
    # 原图的行,列,通道数
    high, width, channel = src.shape
    # 创建一个空的目标图像
    dst_img = np.zeros((dsize[1], dsize[0], channel), src.dtype)
    for c in range(channel):
        for i in range(high):
            for j in range(width):
                Z = M[2][0] * j + M[2][1] * i + M[2][2]
                # 当前原图x坐标,对应目标图像位置
                dst_x = int((M[0][0] * j + M[0][1] * i + M[0][2]) / Z + 0.5)
                # 当前原图y坐标,对应目标图像位置
                dst_y = int((M[1][0] * j + M[1][1] * i + M[1][2]) / Z + 0.5)
                if 0 <= dst_x < dsize[0] and 0 <= dst_y < dsize[1]:
                    # 将原图该位置的灰度值赋予目标图像
                    dst_img[dst_y, dst_x, c] = src[i, j, c]
    print("返回结果")
    return dst_img


if __name__ == '__main__':
    # 原图
    src_img = cv2.imread("photo1.jpg")
    # 原图像四个点坐标
    src_matrix = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    # 目标图像四个点坐标
    dst_matrix = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpMatrix = getWarpMatrix(src_matrix, dst_matrix)
    print("warpMatrix \n", warpMatrix)
    dsize = (337, 488)
    dst_img = warp_Perspective(src_img, warpMatrix, dsize)
    cv2.imshow("src_img", src_img)
    cv2.imshow("dst_img", dst_img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
