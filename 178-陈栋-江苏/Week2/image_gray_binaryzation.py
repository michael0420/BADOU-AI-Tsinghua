"""
@author: Dong Chen
@time: 01/11/2022
@Reference: Teacher Wang's code

change color image gray and binary conversion
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

#gray
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#binaryzation
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
print(img_binary)
print(img_binary.shape)


