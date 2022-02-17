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

import cv2 as cv
print(cv.__version__)
import imutils

img = cv.imread("D:/BaiduNetdiskDownload/photo1.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
dilate = cv.dilate(blurred, cv.getStructuringElement(cv.MORPH_RECT, (1,1)))

#里面函数的返回值就是 kernnel shape=【11，11】, 值为1
print(cv.getStructuringElement(cv.MORPH_RECT, (11, 11)))
cv.imshow("bule", blurred)
cv.imshow('dial', dilate)
cv.waitKey(0)
#
# edged = cv.Canny(dilate, 30, 120, 3)
# cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #轮廓检测
#
# #判断opencv的版本
# if imutils.is_cv2():
#     cnts = cnts[0]
# else:
#     cnts = cnts[1]
#
# docCnt = None
# if len(cnts) > 0:
#     cnts = sorted(cnts, key=cv.contourArea, reverse=True) #根据轮廓面积从大到小排序
#     for c in cnts:
#         peri = cv.arcLength(c, True)
#         approx = cv.approxPolyDP(c, 0.02*peri, True)
#         # 轮廓为 4个点表示找到纸张
#         if len(approx) == 4:
#
#             docCnt = approx
#             break
# for peak in docCnt:
#     peak = peak[0]
#     cv.circle(img, tuple(peak), 10, (255, 0, 0))
#
# cv.imshow('img', img)
# cv.waitKey(0)

