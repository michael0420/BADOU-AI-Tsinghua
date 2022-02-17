import cv2
import numpy as np


img = cv2.imread('lenna.png')

b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
intb, intg, intr = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)

gray_fun1 = r*0.3 + g*0.59 + b*0.11
gray_fun1 = gray_fun1.astype(np.uint8)

gray_fun2 = (intr*30 + intg*59 + intb*11)/100
gray_fun2 = gray_fun2.astype(np.uint8)

gray_fun3 = (intr*76 + intg*151 + intb*28)>>8
gray_fun3 = gray_fun3.astype(np.uint8)
gray_fun4 = (r + g + b)/3
gray_fun4 = gray_fun4.astype(np.uint8)
gray_fun5 = g

gray_mask = gray_fun1>144
gray_2valimg = gray_mask*255
gray_2valimg = gray_2valimg.astype(np.uint8)

cv2.imshow('gray_fun1', gray_fun1)
cv2.imshow('gray_fun2', gray_fun2)
cv2.imshow('gray_fun3', gray_fun3)
cv2.imshow('gray_fun4', gray_fun4)
cv2.imshow('gray_fun5', gray_fun5)
cv2.imshow('gray_2valimg', gray_2valimg)
cv2.waitKey(0)
