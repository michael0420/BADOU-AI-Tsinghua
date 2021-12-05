import cv2 #pip install opencv-python
import numpy as np
def function(img, zoomx, zoomy):
    height, width, channels = img.shape
    emptyImage = np.zeros((zoomx, zoomy, channels), np.uint8)
    sh = zoomx/height
    sw = zoomy/width
    for i in range(zoomx):
        for j in range(zoomy):
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i,j] = img[x,y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom = function(img, 800, 800)
print(zoom)
print(zoom.shape)

cv2.imshow("nearest interp:", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
