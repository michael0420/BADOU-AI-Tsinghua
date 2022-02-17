#coding:utf-8

import cv2 as cv
from skimage import util

'''
skimage中的util.random_noise函数
'''

img = cv.imread('lenna.png')
img_noise = util.random_noise(img, mode='gaussian')

cv.imshow('img', img)
cv.imshow('img_noise', img_noise)
cv.waitKey(0)
cv.destroyWindow('all')
