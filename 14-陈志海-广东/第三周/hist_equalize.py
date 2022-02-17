#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 对单通道图像进行直方图均衡化
def hist_equalize(src, color='r', if_draw=True):
    hist_src = cv2.calcHist([src], [0], None, [256], [0, 256])
    # 计算各灰度等级的累计概率
    prop_sum = np.zeros(256, dtype=np.float32)
    prop_sum[0] = hist_src[0] / src.size
    for i in range(1, hist_src.shape[0]):
        prop_sum[i] = prop_sum[i-1] + hist_src[i][0] / src.size
    h, w = src.shape
    result = np.zeros(src.shape, dtype=src.dtype)
    for i in range(h):
        for j in range(w):
            gray_level = src[i, j]
            result[i, j] = prop_sum[gray_level] * 256 - 1

    if if_draw:
        hist_res = cv2.calcHist([result], [0], None, [256], [0, 256])
        plt.figure("equalize, channel="+color)
        plt.subplot(221)
        plt.imshow(src, cmap='gray')
        plt.title("src")
        plt.axis("off")
        plt.subplot(222)
        plt.plot(hist_src, color=color)
        plt.title("hist_src")
        plt.subplot(223)
        plt.imshow(result, cmap='gray')
        plt.title("result")
        plt.axis("off")
        plt.subplot(224)
        plt.plot(hist_res, color=color)
        plt.title("hist_result")
        # plt.show()
    return result


# main
img_bgr = cv2.imread("lenna.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
color = ('b', 'g', 'r')
img_r = hist_equalize(img_rgb[:, :, 0], color='r', if_draw=True)
img_g = hist_equalize(img_rgb[:, :, 1], color='r', if_draw=False)
img_b = hist_equalize(img_rgb[:, :, 2], color='r', if_draw=False)
img_out = cv2.merge([img_r, img_g, img_b])

plt.figure("equalize, channel=[r,g,b]")
plt.subplot(221)
plt.imshow(img_rgb)
plt.title("lenna")
plt.axis('off')
plt.subplot(222)
plt.imshow(img_out)
plt.title("lenna_hist_equalize")
plt.axis('off')
plt.show()
