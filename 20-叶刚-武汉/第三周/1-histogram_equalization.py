"""
@GiffordY
histogram equalization algorithm
"""

import cv2 as cv
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def equalize_hist_gray(src_image):
    ret = src_image.shape
    assert len(ret) == 2
    dst_image = np.zeros((ret[0], ret[1]), dtype=np.uint8)
    total_num = ret[0] * ret[1]
    nk = dict(Counter(src_image.flatten()))
    nk = dict(sorted(nk.items(), key=lambda x: x[0]))
    pr_k = dict((k, v / total_num) for k, v in nk.items())
    pr_k_sum = dict()
    sum = 0.0
    for k, v in pr_k.items():
        sum = sum + v
        pr_k_sum[k] = sum

    mapping = dict((k, max(min(round(v * 256 - 1), 255), 0)) for k, v in pr_k_sum.items())
    # mapping = dict()
    # for k, v in pr_k_sum.items():
    #     val = min(round(v*256-1), 255)
    #     val = max(val, 0)
    #     mapping[k] = val

    for row in range(ret[0]):
        for col in range(ret[1]):
            dst_image[row, col] = mapping[src_image[row, col]]
    return dst_image


def equalize_hist_color(src_image, mode='YUV'):
    ret = src_image.shape
    if len(ret) == 2:
        dst_image = equalize_hist_gray(src_image)
    else:
        if mode == 'RGB':
            dst_image = np.zeros((ret[0], ret[1], ret[2]), np.uint8)
            img_b, img_g, img_r = cv.split(src_image)
            img_b_equ = equalize_hist_gray(img_b)
            img_g_equ = equalize_hist_gray(img_g)
            img_r_equ = equalize_hist_gray(img_r)
            cv.merge([img_b_equ, img_g_equ, img_r_equ], dst_image)
        else:
            img_yuv = cv.cvtColor(src_image, cv.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = equalize_hist_gray(img_yuv[:, :, 0])
            dst_image = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
        print('Histogram equalization of color image based on {} color space'.format(mode))
    return dst_image


if __name__ == '__main__':
    img = cv.imread('lenna.png')
    img_equ_my = equalize_hist_color(img, mode='YUV')
    # cv.imshow('img', img)
    # cv.imshow('img_equ_my', img_equ_my)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray_equ_my = equalize_hist_gray(img_gray)
    img_gray_equ_cv = cv.equalizeHist(img_gray)
    # cv.imshow('img_gray', img_gray)
    # cv.imshow('img_gray_equ_my', img_gray_equ_my)
    # cv.imshow('img_gray_equ_cv', img_gray_equ_cv)

    # Show results: Histogram equalization for color image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('src image')
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.title('dst image')
    img_equ_my_rgb = cv.cvtColor(img_equ_my, cv.COLOR_BGR2RGB)
    plt.imshow(img_equ_my_rgb)
    plt.show()

    # Show results: Histogram equalization for grayscale image
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('src gray image')
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('dst image my')
    plt.imshow(img_gray_equ_my, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('dst image cv')
    plt.imshow(img_gray_equ_cv, cmap='gray')
    plt.show()

    # Show results: Histogram
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('img_gray_equ_my')
    plt.imshow(img_gray_equ_my, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('histogram_my')
    plt.hist(img_gray_equ_my.ravel(), 256, (0, 255), True)
    plt.subplot(2, 2, 3)
    plt.title('img_gray_equ_cv')
    plt.imshow(img_gray_equ_cv, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title('histogram_cv')
    plt.hist(img_gray_equ_cv.ravel(), 256, (0, 255), True)
    plt.show()

    cv.waitKey(0)

