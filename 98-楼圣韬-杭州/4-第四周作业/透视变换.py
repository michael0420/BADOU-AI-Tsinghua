import cv2
import numpy as np


# 寻找图像的顶点
from numpy import single


def findpoint(img):  # img为一张灰度图
    blurred = cv2.GaussianBlur(img, (7, 7), 0)  # 高斯模糊，降噪
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    '''
    dilate()函数可以对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中领域的形状,各点像素值将被替换为对应邻域上的最大值
    dst(x,y)=max sre(x+x',y+y')
    cv2.dilate(img,kernel) img为输入图像，kernel为膨胀操作结构元素，此处用cv2.getStructuringElement创建
    cv2.getStructuringElement(a,(x,y)) a 结构类型(十字，圆，多边形)，(x,y)结构大小
    '''

    edged = cv2.Canny(dilate, 30, 120, apertureSize=3)
    # cv2.Canny(img,低阈值，高阈值，apertureSize =soble算子大小) apertureSize 默认为3

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    '''
    cv2.findContours 用于检测物体轮廓，先看cv2.findContours函数原型：
    findContours( InputOutputArray image, OutputArrayOfArrays contours,
                              OutputArray hierarchy, int mode,
                              int method, Point offset=Point())
    
    第一个参数:image，单通道图像矩阵，可以是灰度图，更常用的是二值图，一般是经过Canny拉普拉斯处理过的
    第二个参数(可缺省,存于内部):contours,定义为“vector<vector<Point>>contours” 是一个向量，并且是一个双重向量
       向量内每个元素保存了一组由连续Point点构成的集合的向量，每一组Point点集就是以一个轮廓。
       有多少元素，向量contours就有多少元素
    第三个参数(内部参数，可确省):hierarchy,定义为“vector<Vec4i>hierarchy” 定义了一个向量内每一个元素包含了4个int型变量
       hierarchy与contours内的元素一一对应，向量的容量相同。
       四个int型变量：后一个轮廓，前一个轮廓，父轮廓，内嵌轮廓的索引编号   
    第四个参数:int型的mode，定义轮廓的检索模式：
       取值1:CV_RETR_EXTERNAL 只检测最外围轮廓，包含再外围轮廓的内围轮廓被忽略
       取值2:CV_RETR_LIST 检索所有轮廓(内围外围)，但是检测到的轮廓彼此之间独立
       取值3:CV_RETR_CCOMP 检测所有的轮廓，但所有轮廓只建立两个等级关系
       取值4:CV_RETR_TREE 检测所有轮廓，所有轮廓建立一个等级树结构
    第五个参数:int型的method，定义轮廓的近似方法:
        取值1:CV_CHAIN_APPROX_NONE 保存物体边界上所有的连续轮廓点到contours向量内
        取值2:CV_CHAIN_APPROX_SIMPLE 仅保存轮廓拐点信息，把所有轮廓拐点出的点存入contours向量内
        取值3和4:CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法
    '''
    cnts = cnts[0]  # 将contours向量取出
    dt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
        for c in cnts:
            peri = cv2.arcLength(c, True)  # 计算轮廓周长
            # cv2.arcLength(InputArray curve, bool closed) curve为轮廓顶点，closed为轮廓是否封闭标识符

            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
            '''
            cv2.approxPolyDP() 多边形逼近
            作用:
            对目标图像进行近似多边形拟合，使用一个较少顶点的多边形去拟合一个曲线轮廓，要求拟合曲线与实际轮廓曲线的距离小于某一阀值。

            函数原形:
            cv2.approxPolyDP(curve, epsilon, closed) -> approxCurve

            参数:
            curve : 图像轮廓点集，一般由轮廓检测得到
            epsilon : 原始曲线与近似曲线的最大距离，参数越小，两直线越接近
            closed : 得到的近似曲线是否封闭，一般为True

            返回值:
            approxCurve : 返回的拟合后的多边形顶点集。
            '''

            # 轮廓为四个点表示找到纸张
            if len(approx) == 4:
                dt = approx
                break

    for peak in dt:
        peak = peak[0]
        cv2.circle(img, tuple(peak), 10, (255, 0, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return dt


def WarpPerspectiveMatrix1(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    # assert 函数，满足条件继续运行程序，不满足条件直接报错
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # 创建矩阵，A*warpMatrix=B
    B= np.zeros((2*nums,1))
    for i in range(nums):  # 按格式存入矩阵
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    A = np.mat(A)  # 用mat创建矩阵，并用A.I返回逆矩阵
    warpMatrix1 = A.I*B  # 求出 a1n(n=1_8)

    # 结果处理
    warpMatrix1 = np.array(warpMatrix1).T[0]
    warpMatrix1 = np.insert(warpMatrix1, warpMatrix1.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix1 = warpMatrix1.reshape((3, 3))
    return warpMatrix1
img = cv2.imread('photo1.jpg')
img1=img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fp=findpoint(gray)
for i in fp:
    print(i)

src1 = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
print(src1)

dst1 = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(dst1)
wt=WarpPerspectiveMatrix1(src1, dst1)
#  也可以直接调用cv2函数接口直接求变换矩阵: m=cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(wt)
res = cv2.warpPerspective(img1,wt,(337,488)) # 透视变换函数，也可手动实现
cv2.imshow("src",img)
cv2.imshow("result",res)
cv2.waitKey(0)

