import cv2
import numpy as np

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]#4
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B   ==>   A-1*A*warpMatrix = A-1 * B  ==>  warpMatrix = A-1 * B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]#每次循环，取一行
        print(A_i)#打印整行
        print(A_i[1])#打印该行第二个元素
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,-A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    A = np.mat(A)#转换成矩阵
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    print(warpMatrix)#列向量
    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]#转置，变成行向量
    print(warpMatrix)
    print(warpMatrix.shape[0])
    #numpy.insert(arr,obj,value,axis=None)  #在向量warpMatrix的，第axis=0(行：0；列：0)维度的第warpMatrix.shape[0]位置插入values=1.0
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    result3 = img.copy()
    #print('warpMatrix')
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])#用寻找顶点算法可找出
    src = np.array(src)
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]]) #校正后图像的四个顶点坐标
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
    result = cv2.warpPerspective(result3, warpMatrix, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)

    cv2.waitKey(0)

'''
#注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
'''