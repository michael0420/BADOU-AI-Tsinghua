import cv2
import numpy as np

def model(img):
    h,w,ch = img.shape
    dst_img = np.zeros((800,800,ch),dtype=np.uint8)
    # 比例关系
    x_rate = 800/h
    y_rate = 800/w
    for i in range(800):
        for j in range(800):
            x = int(i/x_rate)
            y = int(j/y_rate)
            dst_img[i,j] = img[x,y]

    return dst_img

img = cv2.imread('lenna.png')
dst = model(img)
cv2.imshow('dst',dst)
cv2.waitKey(0)