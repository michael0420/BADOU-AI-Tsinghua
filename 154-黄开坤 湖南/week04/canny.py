#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt


img = cv.imread('lenna.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_gray = cv.imread('lenna.png', 0)

img_canny = cv.Canny(img_gray, 200, 300)
cv.imshow('canny_img', img_canny)
cv.waitKey()
cv.destroyWindow('all')
