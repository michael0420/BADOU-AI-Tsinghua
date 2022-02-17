from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("lenna.png")

# Test 1:rgb2gray
# call opencv
img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #call opencv
#
h, w = img.shape[0], img.shape[1]
img_gray = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        tmp = img[i, j]
        img_gray[i, j] = int(tmp[0] * 0.11 + tmp[1] * 0.59 + tmp[2] * 0.3)
print(img_gray[256, 256])
cv2.imshow("lenna", img)
cv2.imshow("lenna gray", img_gray)
cv2.imshow("lenna gray1", img_gray1)
cv2.waitKey(0)

# Test 2: binary image
#call opencv
ret, img_binary1 = cv2.threshold(img_gray1, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("lenna binary", img_binary1)

img_binary = np.where(img_gray >= 127, 255, 0)
img_binary = img_binary.astype(np.uint8)
print(img_binary.dtype)
print(img_binary)
cv2.imshow("lenna binary1", img_binary)
cv2.waitKey(0)