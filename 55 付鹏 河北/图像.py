from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("F:/lenna.png")
h1, w1, ch1 = img.shape
cv2.imshow('Image', img)
# b, g, r = cv2.split(img)
# gray = 0.11*b + 0.59*g + 0.3*r
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 = rgb2gray(img)
# plt.subplot(2, 2, 1)
cv2.imshow('Image1', img1)
# h, w = img1.shape
# print(h, w)
# i = 0
# j = 0
# for i in range(h):
#     for j in range(w):
#         if img1[i, j] <= 126:
#             img1[i, j] = 0
#         else:
#             img1[i, j] = 255
# cv2.imshow('Img11', img1)

# (a, img22) = cv2.threshold(img1, 0.4, 1, cv2.THRESH_BINARY)
img22 = img1
h, w = img22.shape
for i in range(h):
    for j in range(w):
        if img22[i, j] <= 0.45:
            img22[i, j] = 1
        else:
            img22[i, j] = 0
cv2.imshow('Img22', img22)
print(h1, w1)
cv2.waitKey(0)
