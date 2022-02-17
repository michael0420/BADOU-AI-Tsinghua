# encoding: utf-8

import cv2
import numpy as np


def canny_detect(lowThreshold):
    detect_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
    detect_edges = cv2.Canny(detect_edges, lowThreshold, lowThreshold * radio, apertureSize=3)  # Canny
    dst = cv2.bitwise_and(img, img, mask=detect_edges)
    cv2.imshow('canny demo', dst)


lowThreshold = 0
max_lowThreshold = 100
radio = 3

img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('canny demo')

# 设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
1是trackbar对象的名字; 2是这个trackbar对象所在面板的名字;3这个trackbar的默认值,也是调节的对象;4是这个trackbar上调节的范围(0~count;5是调节trackbar时调用的回调函数名
'''
cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold,
                   canny_detect)  # 注意此处为单一参数函数， 也就是说回调的函数必须是一个参数的
canny_detect(0)  # initialization

cv2.waitKey(0)
cv2.destroyAllWindows()
