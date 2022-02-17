from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#最近邻插值
img = cv2.imread("lenna.png")
dst_w = 800
dst_h = 800
scale_w = float(dst_w / img.shape[1])
scale_h = float(dst_h / img.shape[0])
img_dst = np.zeros([dst_h, dst_w, img.shape[2]], np.uint8)
for i in range(dst_h):
    for j in range(dst_w):
        ori_x = int(j / scale_w)
        ori_y = int(i / scale_h)
        img_dst[i, j] = img[ori_y, ori_x]

cv2.imshow("img", img)
cv2.imshow("img_interpolation", img_dst)
cv2.waitKey(0)

#双线性插值
img_dst1 = np.zeros([dst_h, dst_w, img.shape[2]], np.uint8)
for i in range(dst_h):
    for j in range(dst_w):
        ori_x = (j + 0.5) / scale_w - 0.5
        ori_y = (i + 0.5) / scale_h - 0.5
        src_x0 = int(np.floor(ori_x))
        src_x1 = min(src_x0 + 1, img.shape[1] - 1)
        src_y0 = int(np.floor(ori_y))
        src_y1 = min(src_y0 + 1, img.shape[0] - 1)
        for k in range(3):
            tmp0 = (src_x1 - ori_x) * img[src_y0, src_x0, k] + (ori_x - src_x0) * img[src_y0, src_x1, k]
            tmp1 = (src_x1 - ori_x) * img[src_y1, src_x0, k] + (ori_x - src_x0) * img[src_y1, src_x1, k]
            img_dst1[i, j, k] = int((src_y1 - ori_y) * tmp0 + (ori_y - src_y0) * tmp1)

cv2.imshow("interpolation2", img_dst1)
cv2.waitKey(0)
