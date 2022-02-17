"""
@author: Dong Chen
@time: 01/11/2022
@Reference: Teacher Wang's code

do image nearest interpolation
"""

import cv2
import numpy as np

def nearest_interp(img):
    height,width,channels = \
        img.shape
    des_height = 800
    des_width = 800
    sh = des_height/height
    sw = des_width/width
    emptyImage=np.zeros((des_height,des_width,channels),np.uint8)
    for i in range(des_height):
        for j in range(des_width):
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage

if __name__ == '__main__':
    img=cv2.imread("lenna.png")
    zoom = nearest_interp(img)
    cv2.imshow("original image", img)
    cv2.imshow("nearest interpolation", zoom)
    cv2.waitKey(0)
    cv2.destroyWindow()