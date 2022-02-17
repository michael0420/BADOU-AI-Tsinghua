#coding:utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt

def nearestInterp(img):
    hei, wid, cha = img.shape
    img_enpty = np.zeros((800, 800, cha), dtype=np.uint8)
    for i in range(800):
        for j in range(800):
            x = int(i * (hei/800))      #取整
            y = int(j * (wid/800))
            img_enpty[i, j] = img[x, y]
    return img_enpty

img= cv2.imread('lenna.png')
zoom = nearestInterp(img)
print(img.shape, zoom.shape)
cv2.imshow('nearest interpolation', zoom)
cv2.imshow('img', img)
cv2.waitKey(0)
