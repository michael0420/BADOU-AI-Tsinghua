from skimage.color import rgb2gray #conda install scikit-image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 #pip install opencv-python
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0]*0.11 +m[1]*0.59 +m[2]*0.3)
        #print("m", img[i, j]) #3值
        #print("img_gray", img_gray[i, j]) #1值
#print(img)
#print(img_gray)
#cv2.imshow("image show gray", img_gray)
#cv2.imshow("image show origin",img)
#print(img.shape)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("----image lenna------")
print(img)
#灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray------")
print(img_gray)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-------img_binary---------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()
plt.subplot()
#cv2.waitKey(0)
