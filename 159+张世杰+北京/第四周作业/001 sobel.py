# encoding: utf-8
import cv2
import numpy as np


img = cv2.imread('lenna.png', 0)  # 0 灰度图像， 1 彩色图像
'''
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。
'''
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
'''
此时生成的图片无法显示， 需要通过cv2.convertScaleAbs()函数，进行转换：转换原理是取绝对值后截断
'''
abs_x = cv2.convertScaleAbs(x)
abs_y = cv2.convertScaleAbs(y)
'''
将两个图像叠加：cv2.addweight
'''
dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
dst_tg = np.hstack((abs_x, abs_y, dst))

cv2.imshow('img', dst_tg)
cv2.waitKey(0)
cv2.destroyAllWindows()
