import math
import cv2
import numpy as np
import random


def gaussi(mean, sigma, x, y):
    random_gaussi = np.zeros((x, y))
    t1 = 1 / (2 * math.pi * sigma**2)
    t2 = -2*sigma**2
    for i in range(x):
        for j in range(y):
            t3 = ((i - mean)**2 + (j - mean)**2) / t2
            random_gaussi[i, j] = t1 * math.exp(t3)
    return random_gaussi


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    cv2.imshow('src_img', img_gray)
    for i in range(h):
        for j in range(w):
            img_gray[i, j] =img_gray[i, j] + random.gauss(22, 2)
    cv2.imshow('noise_img', img_gray)
    cv2.waitKey(0)
