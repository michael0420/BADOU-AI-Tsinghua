import cv2
import numpy as np

srcImg = cv2.imread("lenna.png")
sh, sw = srcImg.shape[:2]
dh = 1000
dw = 1000
destImg = np.zeros((dh, dw, 3), np.uint8)
scaleH = dh/sh
scaleW = dw/sw
for i in range(dh):
    for j in range(dw):
        destImg[i, j] = srcImg[int(i/scaleH), int(j/scaleW)]
cv2.imshow("origin", srcImg)
cv2.imshow("nearest", destImg)
cv2.waitKey()




