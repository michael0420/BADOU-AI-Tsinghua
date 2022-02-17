# coding: utf-8
# Author：Jason
# Date ：2021/11/25 9:55 下午
# Tool ：PyCharm
import numpy as np


def img_2_gray(img):
    h, w, c = img.shape
    if c == 1:
        return img

    gray_img = np.zeros((h, w, 1))

    for i in range(h):
        for j in range(w):
            g, b, r = img[i, j][:3]
            gray_img[i, j] = int(g * 0.11 + b * 0.59 + r * 0.3)  # RGB 0.3, 0.59, 0.11 ,cv2 is GBR
    return gray_img


def img_2_binary(img, threshold, pix1, pix2):

    def _compare(c_pix):  # 对当前像素和阈值进行对比，比较过后返回对应像素
        return pix1 if c_pix >= threshold else pix2

    h, w, c = img.shape[:2]
    binary_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            pixes = img[i, j]
            if len(pixes) == 1:  # 兼容单通道和RGB，TODO：未考虑RGBA
                binary_img[i, j] = _compare(pixes)
            else:
                binary_img[i, j] = list(map(pixes, _compare))  # 对三个通道的像素都做二值化处理

    return binary_img
