# This is a Python implementation script of generating gray image.

# import library
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# read a picture
img = cv2.imread("AWACS.jpeg")

# get size of the pic
height, width =img.shape[:2]

# create a new single-channel pic with same size
img_gray = np.zeros([height, width], img.dtype)

# gray level calculation
for i in range(height):
    for j in range(width):
        bgr = img[i,j]
        img_gray[i, j] = int(bgr[1])
#        img_gray[i,j] = int(bgr[0] + bgr[1] + bgr[2])/3
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)
cv2.waitKey(2000)