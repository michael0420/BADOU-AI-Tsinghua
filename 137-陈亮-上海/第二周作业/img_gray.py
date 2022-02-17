import cv2
import numpy as np

def img_gray(img):
    h,w,ch = img.shape
    dst_img = np.zeros([h,w],dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            dst_img[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

    return dst_img

#第一种方式
img = cv2.imread('lenna.png')
gray = img_gray(img)
cv2.imshow('gray',gray)
cv2.waitKey(0)
# 第二种方式
img = cv2.imread('lenna.png')
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray2',gray)
cv2.waitKey(0)