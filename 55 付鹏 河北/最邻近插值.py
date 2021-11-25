import cv2
import numpy as np

def function(image):
    imag1 = np.zeros([800, 800, channels], dtype=np.uint8)
    for i in range(800):
        for j in range(800):
            x = int((w/800)*i)
            y = int((h/800)*j)
            imag1[i, j] = image[x, y]
    return imag1
image = cv2.imread('F:/lenna.png')
cv2.imshow('image1', image)
h, w, channels = image.shape
image11 = function(image)
cv2.imshow('image11', image11)

cv2.waitKey(0)