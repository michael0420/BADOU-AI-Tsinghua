"""
author:xswwhy
最邻近插值
优点:简单
缺点:灰度上不连续,出现锯齿
"""
import cv2
import numpy as np


def nearest_interp(src_img, dst_h, dst_w):
    src_h, src_w, c = src_img.shape
    dst_img = np.zeros([dst_h, dst_w, c], src_img.dtype)
    scale_h = dst_h / src_h
    scale_w = dst_w / src_w
    for i in range(dst_h):
        for j in range(dst_w):
            h = int(i / scale_h)
            w = int(j / scale_w)
            dst_img[i, j] = src_img[h, w]
    return dst_img


img_bgr = cv2.imread("lenna.png")
img_large = nearest_interp(img_bgr, 1000, 1000)
img_small = nearest_interp(img_bgr, 200, 200)

# show
cv2.imshow("original", img_bgr)
cv2.imshow("lager", img_large)
cv2.imshow("small", img_small)
cv2.waitKey(0)
