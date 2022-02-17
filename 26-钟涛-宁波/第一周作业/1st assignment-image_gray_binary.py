"""
author： 26-钟涛-宁波
图像灰度化以及二值化
"""

import cv2
import matplotlib.pyplot as plt

"""
图像灰度化
"""

im_src= cv2.imread("lenna.png");
import numpy as np;
h,w= im_src.shape[:2];

im_gray = np.zeros([h,w],im_src.dtype);
for i in range(h):
    for j in range(w):
        im = im_src[i,j];
        im_gray[i,j] = im[0]*0.11 +im[1]*0.59+im[2]+0.3; #灰度化 BGR
print(im_gray);

"""图像二值化"""
im_binary = np.zeros([h,w],im_gray.dtype);
for i in range(h):
    for j in range(w):
        if((im_gray[i,j]/255)<0.5):
            im_binary[i,j]=0;
        else:
            im_binary[i,j] = 1;
print(im_binary);

plt.subplot(221);
plt.imshow(im_src);

plt.subplot(222);
plt.imshow(im_gray,cmap='gray');

plt.subplot(223);
plt.imshow(im_binary,cmap='binary');
plt.show();



