# import lib
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Make pic gray
img = cv2.imread("lenna.png")
# print(img)
# Get the numbers of row and col for the image
row, col = np.shape(img)[:-1]
# Create a new zero-value matrix with the same size of the image
grey = np.zeros([row, col], img.dtype)
# for r in range(row):
#     for c in range(col):
#         ig = img[r, c]
#         grey[r, c] = int(ig[0] * 0.11 + ig[1] * 0.59 + ig[2] * 0.3)

# Cautious: BGR in order for cv2, instead of RGB
im_grey = np.array([[int(img[r, c][0] * 0.11 + img[r, c][1] * 0.59 + img[r, c][2] * 0.3)
                     for c in range(col)] for r in range(row)], img.dtype)
print("image show im_grey: %s" % im_grey)
# cv2.imshow("image show im_grey", im_grey)
# cv2.waitKey()

# Make a pic tow_values
plt.subplot(221)
img1 = plt.imread("lenna.png")
# print(img1)
# im_grey1 = np.array([[img1[r, c][0]*0.3+img1[r, c][1]*0.59+img1[r, c][2]*0.11
#                       for c in range(col)] for r in range(row)], img1.dtype)
# print("image show im_grey1: %s" % im_grey1)
plt.imshow(img1)

# To turn gray with plt
im_grey1 = rgb2gray(img1)
print("image show in im_grey1: %s" % im_grey1)
plt.subplot(222)
plt.imshow(im_grey1, cmap='gray')

# To be two_values with plt
# im_grey2 = np.array([[1 if im_grey1[r, c] > 0.5 else 0
#                       for c in range(col)] for r in range(row)], img1.dtype)

# if figure in im_grey1 > 0.5, figure to be 1. Else, be 0
im_grey2 = np.where(im_grey1 > 0.5, 1, 0)
print("image show in im_grey2: %s" % im_grey2)
plt.subplot(223)
plt.imshow(im_grey2, cmap='gray')
plt.show()
