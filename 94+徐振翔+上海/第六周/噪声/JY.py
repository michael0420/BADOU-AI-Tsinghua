# 目标：手写椒盐噪声算法

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


def jy(img, SNR=0.2):
    """
    :param img: 输入图像
    :param SNR: 信噪比
    :return: 加过噪声的图像
    """
    noise_img = img.copy()
    x = img.shape[0]
    y = img.shape[1]
    # 计算计划填充的椒盐噪声总数
    num = int(x * y * SNR)

    for i in range(num):
        # 随机生成添加噪声的点
        rx = math.floor(x * random.random())
        ry = math.floor(y * random.random())
        # 随机选择椒或盐噪声
        jy0 = random.random()
        if jy0 > 0.5:
            noise_img[rx, ry] = 255
        else:
            noise_img[rx, ry] = 0
    return noise_img


img = cv2.imread('lenna.png')
# img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img1 = jy(img, 0.05)
# img1 = jy(img, 0.3)
# img1 = jy(img)
cv2.imshow('src', img)
cv_show(img1, "jy_img ")
