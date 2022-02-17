import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2*nums, 8))
    B = np.zeros((2*nums, 1))

    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]

    A = np.mat(A)
    warpMatrix = A.I * B
    print(warpMatrix)
    warpMatrix = np.array(warpMatrix).T[0]
    print(warpMatrix)
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    print(warpMatrix)
    warpMatrix = warpMatrix.reshape((3, 3))
    print(warpMatrix)
    return warpMatrix

if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    result3 = img.copy()
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    m = WarpPerspectiveMatrix(src, dst)
    print("warpMatrix:", m)

    result = cv2.warpPerspective(result3, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)

