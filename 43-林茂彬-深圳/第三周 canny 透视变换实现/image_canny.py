#!/usr/bin/env python
# encoding=gbk

'''
Canny边缘检测：优化的程序
'''
import cv2
import numpy as np


'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
apertureSize：可选参数，Sobel算子的大小
'''


#
class image_canny:
    def __init__(self,X,lowThreshold,max_lowThreshold):#初始化实例属性
        self.X = X
        self.lowThreshold=lowThreshold
        self.max_lowThreshold=max_lowThreshold
        self.image_gray = self._image_gray()
        self.image_threshold = self._image_threshold()
        self.image_canny = self._image_canny()
        cv2.imshow('canny', self.image_canny)
        cv2.waitKey(0)

    def _image_gray(self):
        print(self.X)
        image_gray = cv2.cvtColor(self.X, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
        return image_gray


    def _image_threshold(self):#高斯滤波
        print(self.image_gray)
        image_threshold=cv2.GaussianBlur(self.image_gray,(3,3),0)
        return image_threshold

    def _image_canny(self):
        image_canny = cv2.Canny(self.image_threshold,self.lowThreshold,self.max_lowThreshold)  # 第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；第二个参数是滞后阈值1；第三个参数是滞后阈值2。
        return image_canny



if __name__ == '__main__':
    img_source=cv2.imread('lenna.png')
    print(img_source)
    image_canny = image_canny(img_source,0,100)
