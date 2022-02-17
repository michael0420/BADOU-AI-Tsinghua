import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest_interp(img):
    h,w,c = img.shape
    emptyImage = np.zeros((800,800,c),np.uint8)
    sh = 800 / h
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv2.imread(r"lenna.png")
img1 = nearest_interp(img)
# plt.subplot(121)
# plt.imshow(img,cmap='gray')
# plt.subplot(122)
# plt.imshow(img1,cmap='gray')
# plt.show()
cv2.imshow('Origin',img)
cv2.imshow('nearest interp',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
