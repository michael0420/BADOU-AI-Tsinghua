import cv2
import numpy as np
import matplotlib.pyplot as plot

img = cv2.imread("lenna.png")
cv2.imshow("origin", img)

h, w = img.shape[:2]
grayImg = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        grayImg[i, j] = int(0.11 * img[i, j, 0] + 0.59 * img[i, j, 1] + 0.3 * img[i, j, 2])
cv2.imshow("gray", grayImg)

binaryImg = np.where(grayImg >= 127, np.uint8(255), np.uint8(0))
print(binaryImg)
cv2.imshow("binary", binaryImg)
cv2.waitKey()


