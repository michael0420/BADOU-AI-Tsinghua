import numpy as np

'''
求解透视变换矩阵
'''

def WrapPersperctiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4 # 原矩阵和目标矩阵行数一样，并且要大于等于4

    nums = src.shape[0] # 得到矩阵的行数

    A = np.zeros((2 * nums, 8)) # A矩阵，A*wrapMatrix = B
    B = np.zeros((2 * nums, 1)) # B矩阵，只有1列

    # 构建A、B两个矩阵
    for i in range(0, nums):
        A_i = src[i, :]     # 提取每对的数据
        B_i = dst[i, :]

        # 按照公式填充矩阵
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]

        B[2 * i] = B_i[0]
        B[2 * i + 1] = B_i[1]

    # 将A转换为矩阵
    A = np.mat(A)

    # 使用A和B计算出透视变换矩阵，公式：warpMatrix = A.I * B
    warpMatrix = A.I * B # 求出透视变换矩阵，也就是a11、a12、a13等

    warpMatrix = np.array(warpMatrix).T[0] # 得到转置矩阵，转换为1xn的1维矩阵
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) # 按行，在最末尾插入一个1
    warpMatrix = warpMatrix.reshape((3, 3)) # 转换为3X3矩阵
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WrapPersperctiveMatrix(src, dst)
    print(warpMatrix)

