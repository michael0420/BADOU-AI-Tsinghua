import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img=cv2.imread("lenna.png")
length=img.shape[0]
width=img.shape[1]
img_gray = np.zeros([length,width],img.dtype)
img_gray = rgb2gray(img)
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.imshow(img_binary, cmap='gray')
plt.show()