"""
@author: 陈志海
fcn: resize the image with nearest_interpolation method
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def nearest_interp(img_src, size_out=(10, 10)):
    h_in, w_in, c_in = img_src.shape
    h_out, w_out = size_out
    ratio_h = h_out / h_in
    ratio_w = w_out / w_in
    img_out = np.zeros((h_out, w_out, c_in), img_src.dtype)
    for i in range(h_out):
        for j in range(w_out):
            i_in = round(i / ratio_h)
            j_in = round(j / ratio_w)
            if i_in == h_in:
                i_in -= 1
            if j_in == w_in:
                j_in -= 1
            img_out[i, j] = img_src[i_in, j_in]
    return img_out


# img_src
img_src = cv2.imread("lenna.png")
img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
plt.subplot(131)
plt.imshow(img_src)
plt.title("img_src")

# img_resize 1
size_out = (1000, 1000)
img_out = nearest_interp(img_src, size_out)
plt.subplot(132)
plt.imshow(img_out)
plt.title("img_out: size=(%d, %d)" % (size_out[0], size_out[1]))

# img_resize 2
size_out = (100, 100)
img_out = nearest_interp(img_src, size_out)
plt.subplot(133)
plt.imshow(img_out)
plt.title("img_out: size=(%d, %d)" % (size_out[0], size_out[1]))
plt.show()
