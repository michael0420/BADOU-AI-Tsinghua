"""
author: 陈志海
fcn:
    1. convert the  rgb image to gray
    2. convert the gray image to binary
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

# img_rgb
img_bgr = cv2.imread("lenna.png")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)
plt.title("lenna")
plt.axis('off')

# img_gray
img_gray = np.zeros(img.shape, dtype=img.dtype)
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        # gray =  R*0.299 + G*0.587 + B*0.114
        img_gray[i, j] = img[i, j][0] * 0.299 + img[i, j][1] * 0.587 + img[i, j][2] * 0.114
plt.subplot(222)
plt.imshow(img_gray)
plt.title("lenna_gray")
plt.axis('off')

# img_binary
img_bin = np.where(img_gray > 128, 255, 0)
plt.subplot(223)
plt.imshow(img_bin)
plt.title("lenna_binary")
plt.axis('off')

plt.show()
