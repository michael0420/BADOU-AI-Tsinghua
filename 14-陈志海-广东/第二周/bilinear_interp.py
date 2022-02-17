"""
author: 陈志海
fcn1: resize the image with bi_linear_interpolation method
fcn2: resize the image with nearest_interpolation method
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil, floor


def bi_linear_interp(img_src, size_out=(100, 100)):
    h_in, w_in, c_in = img_src.shape
    h_out, w_out, = size_out
    ratio_h = h_out / h_in
    ratio_w = w_out / w_in
    img_out = np.zeros((h_out, w_out, c_in), dtype=img_src.dtype)
    for i in range(h_out):
        for j in range(w_out):
            x = (j + 0.5) / ratio_w - 0.5
            y = (i + 0.5) / ratio_h - 0.5
            x1 = floor(x)
            if x1 == w_in - 1:
                x1 -= 1
            y1 = floor(y)
            if y1 == h_in - 1:
                y1 -= 1
            x2 = x1 + 1
            y2 = y1 + 1
            # ---按公式计算
            # f_r1 = (x2 - x) / (x2 - x1) * img_src[y1, x1] + (x - x1) / (x2 - x1) * img_src[y1, x2]
            # f_r2 = (x2 - x) / (x2 - x1) * img_src[y2, x1] + (x - x1) / (x2 - x1) * img_src[y2, x2]
            # img_out[i, j] = (y2 - y) / (y2 - y1) * f_r1 + (y - y1) / (y2 - y1) * f_r2
            # 按权重计算
            u = x - x1
            v = y - y1
            img_out[i, j] = (1-u) * (1-v) * img_src[y1, x1] + (1-u) * v * img_src[y2, x1] + \
                            u * (1-v) * img_src[y1, x2] + u * v * img_src[y2, x2]
    return img_out


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


img_src = cv2.imread("lenna.png")
img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

plt.figure("lenna")
plt.imshow(img_src)
plt.title("img_src")

plt.figure("lenna_缩小")
size_out = (100, 100)
img_bi_linear = bi_linear_interp(img_src, size_out)
img_nearest = nearest_interp(img_src, size_out)
plt.subplot(121)
plt.imshow(img_bi_linear)
plt.title("lenna_bi_linear")
plt.subplot(122)
plt.imshow(img_nearest)
plt.title("lenna_nearest")

plt.figure("lenna_放大")
size_out = (1000, 1000)
img_bi_linear = bi_linear_interp(img_src, size_out)
img_nearest = nearest_interp(img_src, size_out)
plt.subplot(121)
plt.imshow(img_bi_linear)
plt.title("lenna_bi_linear")
plt.subplot(122)
plt.imshow(img_nearest)
plt.title("lenna_nearest")
plt.show()
