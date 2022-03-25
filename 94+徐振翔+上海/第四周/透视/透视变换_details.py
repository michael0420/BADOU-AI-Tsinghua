# 目标：手写实现透视变换的算法，
# 调用顶点检测接口，
# 使用官方（a_type = 1）或手写的库(a_type = 2)，
# 使用指定（b_type = 1）或顶点检测得到的顶点（b_type = 2）为输入进行图像转换

import cv2
import numpy as np
import imutils
import sys

a_type = 2
b_type = 2


def cv_show(im, name="demo"):
    import cv2
    import numpy as np
    if (type(name) != str) or (type(im) != np.ndarray):
        return
    # show image
    cv2.imshow(name, im)
    # wait a key to destroy window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
cv2.approxPolyDP() 多边形逼近
作用:
对目标图像进行近似多边形拟合，使用一个较少顶点的多边形去拟合一个曲线轮廓，要求拟合曲线与实际轮廓曲线的距离小于某一阀值。

函数原形：
cv2.approxPolyDP(curve, epsilon, closed) -> approxCurve

参数：
curve ： 图像轮廓点集，一般由轮廓检测得到
epsilon ： 原始曲线与近似曲线的最大距离，参数越小，两直线越接近
closed ： 得到的近似曲线是否封闭，一般为True

返回值：
approxCurve ：返回的拟合后的多边形顶点集。
'''


def get_point(im):
    """
    :param im: cv2，BGR彩色图
    :return: 4个顶点坐标
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    # def GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 膨胀处理
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edged = cv2.Canny(dilate, 30, 120, 3)  # 边缘检测，canny算法
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cnts = cnts[0]
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
            # 轮廓为4个点表示找到纸张
            if len(approx) == 4:
                # 记录轮廓
                docCnt = approx
                break

    # 在输入图像对应位置画圈
    for peak in docCnt:
        peak = peak[0]
        cv2.circle(im, tuple(peak), 10, (255, 0, 0))

    return np.float32(docCnt.reshape(4, 2))


def getPerMatrix(src, dst):
    """
    :param src: 转换前源顶点，最少四个
    :param dst: 转换后目标顶点，最少四个
    :return: matrix: 转换矩阵
    """
    if src.shape[0] == dst.shape[0] and src.shape[0] >= 4:
        # 根据矩阵运算，将4个顶点数据分别写入到待计算的矩阵中

        nums = src.shape[0]
        # A * W = B
        A = np.zeros((2 * nums, 8))
        B = np.zeros((2 * nums, 1))

        for i in range(0, nums):
            A_i = src[i, :]
            B_i = dst[i, :]
            A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
            B[2 * i] = B_i[0]
            A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
            B[2 * i + 1] = B_i[1]

        # array生成数组，用np.dot()表示矩阵乘积，（*）号或np.multiply()表示点乘, 用np.linalg.inv()表示逆矩阵
        # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
        # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
        a1 = np.array(A)
        a2 = np.mat(A)
        w1 = np.dot(np.linalg.inv(a1), B)
        w2 = a2.I * B
        # print(w1)
        # print(w2)

        # 结果后处理，添加1
        w = w1.T[0]
        # 插入a_33 = 1
        w = np.insert(w, w.shape[0], 1.0, 0)
        # print(w)
        w = w.reshape((3, 3))
        # print(w)
        return w
    else:
        return []


img = cv2.imread('photo1.jpg')
img1 = img.copy()
# cv_show(img)


'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
# src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# print(src.shape)
if b_type == 1:
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
else:
    src = get_point(img1)
    # print(src.shape)
    # print(src)
    # [[207, 151], [16, 603], [344, 732], [518, 283]]
    dst = np.float32([[0, 0], [0, 488], [337, 488], [337, 0]])

print(img.shape)
# 生成透视变换矩阵；进行透视变换
if a_type == 1:
    m = cv2.getPerspectiveTransform(src, dst)
else:
    m = getPerMatrix(src, dst)
    if len(m) < 1:
        print("input err")
        sys.exit()
print("warpMatrix:\n")
print(m)
# 使用透视变换矩阵进行透视变换
# target = cv2.warpPerspective(img1, m, (337, 488))
target = cv2.warpPerspective(img, m, (337, 488))
# cv2.imshow("img", img)
cv2.imshow("img", img1)
cv_show(target)
