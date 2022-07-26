import cv2
import numpy as np

'''
通过自己选的四个点,计算出透视变换的变换矩阵
X     a11 a12 a13      x
Y  =  a21 a22 a23  *   y
Z     a31 a32 a33      1

两边同时除以Z  Z= a31*x + a32*y + a33
X/Z = x' = (a11*x + a12*y + a13)  / (a31*x + a32*y + a33) 
Y/Z = y' = (a21*x + a22*y + a23)  / (a31*x + a32*y + a33)

为了方便计算,令a33 = 1, 重新整理得:
x' = a11*x + a12*y + a13 - a31*x*x' - a32*y*x'
y' = a21*x + a22*y + a23 - a31*x*y' - a32*y*y'
将所选的点带入计算,计算出变换矩阵
'''


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


img = cv2.imread('photo1.jpg')
img_result = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# m = cv2.getPerspectiveTransform(src, dst)
m = WarpPerspectiveMatrix(src, dst)
print(m)

''''
通过计算出的变换矩阵完成图像的透视变换
cv2.warpPerspective(p1, p2, p3)
p1: src image
p2: 计算好的变换矩阵
p3: 图像展示区域,超出原图的部分都是黑色
'''
result = cv2.warpPerspective(img_result, m, (500, 500))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
