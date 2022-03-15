# 目标：手写高斯噪声算法

import cv2
import random
import math


def cv_show(im, name="demo"):
    import cv2
    import numpy as np
    if (type(name) != str) or (type(im) != np.ndarray):
        return
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gs(img, SNR=0.2, means=2.4, sigma=0.8):
    """
    :param img: 输入图像
    :param SNR: 信噪比
    :return: 加过噪声的图像
    """
    noise_img = img.copy()
    x = img.shape[0]
    y = img.shape[1]
    # 计算计划填充的高斯噪声总数
    num = int(x * y * SNR)

    for i in range(num):
        # 随机生成添加噪声的点
        rx = math.floor(x * random.random())
        ry = math.floor(y * random.random())
        # 随机对图像添加高斯噪声
        noise_img[rx, ry] = noise_img[rx, ry] + random.gauss(means, sigma)
        # 范围判断
        for j in range(3):
            if noise_img[rx, ry][j] < 0:
                noise_img[rx, ry][j] = 0
            elif noise_img[rx, ry][j] > 255:
                noise_img[rx, ry][j] = 255
    return noise_img


img = cv2.imread('lenna.png')
img1 = gs(img, 0.05, 2.4, 0.8)
img1 = gs(img, 0.3)
# img1 = gs(img)
cv2.imshow('src', img)
cv_show(img1, "jy_img ")
