#coding:utf8

import numpy as np
import cv2 as cv

img = cv.imread('photo1.jpg')
result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变化矩阵，进行透视变换
M = cv.getPerspectiveTransform(src, dst)    #得到变换矩阵
print('Matrix:\n', M)
result = cv.warpPerspective(result3, M, (337, 488)) #输出变化后的图片大小
cv.imshow('src', img)
cv.imshow('result', result)
cv.waitKey(0)
cv.destroyWindow('all')
