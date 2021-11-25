import numpy as np
import cv2
def function(image):
    v = u = 0.1
    image1 = np.zeros([800, 800, ch], dtype=np.uint8)
    for i in range(800):
        for j in range(800):
            x = int((w/800)*i)
            y = int((h/800)*j)
            if x <= 510 and y <= 510:
                image1[i, j] = (1-u)*(1-v)*image[x, y]+(1-u)*v*image[x, y+1]+u*(1-v)*image[x+1, y]+u*v*image[x+1, y+1]
            else:
                image1[i, j] = image[x, y]
    return image1
image = cv2.imread('F:/lenna.png')
h, w, ch = image.shape
# cv2.imshow('image', image)
a = function(image)

cv2.imshow('a', a)
cv2.waitKey(0)

