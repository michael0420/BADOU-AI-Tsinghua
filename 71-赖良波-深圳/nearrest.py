import cv2
import numpy as np


def fc(img):
    h,w,c=img.shape
    new_img=np.zeros((800,800,c),img.dtype)
    sh=800/h
    sw=800/w
    for i in range(800):
        for j in range(800):
            new_h=int(i/sh)
            new_w=int(j/sw)
            new_img[i,j]=img[new_h,new_w]
    return new_img


img = cv2.imread("lenna.png")

new_img=fc(img)
cv2.imshow("new_img_title",new_img)
cv2.waitKey(0)


