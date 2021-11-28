# This is a Python implementation script of image binary operation.

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

# binary image calculation
img_binary = np.where(img_gray >= 100, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.imshow(img_binary, cmap='gray')
plt.show()
cv2.waitKey(2000)

