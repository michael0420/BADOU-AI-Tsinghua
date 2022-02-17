import cv2
import numpy as np

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
Canny函数原型如下：
    Canny(image, threshold1, threshold2, edges, apertureSize, L2gradient)，函数常用参数含义如下
    image：8bit的原始图像
    threshold1：双阈值算法中的第1个阈值，也就是低阈值
    threshold2：双阈值算法中的第2个阈值，也就是高阈值
    edges：得到的边缘图像
    apertureSize：Sobel算子核大小，也就是Sobel算子矩阵大小
'''
img_canny = cv2.Canny(img_gray, 100, 300)
cv2.imshow("canny", img_canny)
cv2.waitKey(0)