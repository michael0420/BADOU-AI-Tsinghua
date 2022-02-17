import numpy as np
import cv2

def function(img):
    height, width, channels = img.shape   #shape得到包括图像高、宽、通道数量的列表
    emptyImage = np.zeros((1000, 1000, channels), np.uint8)
    th = 1000/height
    tw = 1000/width
    for i in range(1000):
        for j in range(1000):
            x = int(i/th)
            y = int(j/tw)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv2.imread('lenna.png')
z = function(img)
print(z)
print(z.shape)
cv2.imshow('nearest interpolation', z)
cv2.imshow('image', img)
cv2.waitKey(0)
