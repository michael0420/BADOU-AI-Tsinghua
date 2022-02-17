# encoding: utf-8
import cv2
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def near_est(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread('lenna.png')
print(img.shape)
cv_show('img', img)
bigger = near_est(img)
cv_show('big', bigger)
print(bigger.shape)