"""
author:xswwhy
彩色图片灰度化,二值化
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_bgr = cv2.imread("lenna.png")
# 直接使用cv完成灰度化
img_gray_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 手动计算灰度(采用心理学公式) gray = 0.3*R + 0.59*G + 0.11*B
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
h, w = img_rgb.shape[:2]
img_gray_manual = np.zeros([h, w], img_rgb.dtype)
for i in range(h):
    for j in range(w):
        pixel = img_rgb[i, j]
        img_gray_manual[i, j] = int(pixel[0] * 0.3 + pixel[1] * 0.59 + pixel[2] * 0.11)

# 直接使用cv完成二值化,这里阈值取127
_, img_binary_cv2 = cv2.threshold(img_gray_cv2, 127, 255, cv2.THRESH_BINARY)
# 手动二值化
img_binary_manual = np.zeros([h, w], img_rgb.dtype)
for i in range(h):
    for j in range(w):
        if img_gray_cv2[i, j] >= 127:
            img_binary_manual[i, j] = 255
        else:
            img_binary_manual[i, j] = 0

# show
imgs = [img_rgb, img_gray_cv2, img_gray_manual, img_binary_cv2, img_binary_manual]
desc = ["original", "gray-opencv", "gray-manual", "binary-opencv", "binary-manual"]
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 10))
for index, img in enumerate(imgs):
    ax = axs[index]
    ax.axis("off")
    if index == 0:
        ax.imshow(img)
    else:
        ax.imshow(img, "gray")
    ax.text(256, 570, desc[index], fontsize=20, ha="center")
plt.show()
