import cv2
#import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lenna.png')
h,w,c = img.shape
sh=800/h
sw=800/w
dst_img = np.zeros((800, 800, c),np.uint8)
for i in range(800):
    for j in range(800):
        dst_img[i,j] = img[round(i/sh),round(j/sh)]

cv2.imshow("NIPimage",dst_img)
cv2.imshow("image",img)
cv2.waitKey(0)


#plt.subplot(211)
#plt.imshow(img)

#plt.subplot(212)
#plt.imshow(dst_img)
#plt.show()