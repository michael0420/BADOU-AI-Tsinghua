# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/3/28

import numpy as np
import cv2

def mhash(img, shape):
    img_resize = cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    sum = np.sum(gray)
    size = shape[0] * shape[1]
    avg = sum / size
    hash_str = ''   # 哈希值设定为一个字符串
    for i in range(shape[0]):
        for j in range(shape[1]):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else: hash_str = hash_str + '0'
    return hash_str


def dhash(img, shape):
    img_resize = cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    hash_str = ''   # 哈希值设定为一个字符串
    for i in range(shape[1]):
        for j in range(shape[0]-1):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else: hash_str = hash_str + '0'
    return hash_str

def cmp_hash(val1_hash, val2_hash):
    if len(val1_hash) != len(val2_hash):
        return -1
    n = 0
    for i in range(len(val1_hash)):
        if val1_hash[i] != val2_hash[i]:
            n = n + 1
    return n

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
mhash1 = mhash(img1, (8, 8))
mhash2 = mhash(img2, (8, 8))
print('mhash1', mhash1)
print('mhash2', mhash2)
n = cmp_hash(mhash1, mhash2)
print('通过均值哈希计算两张图片的相似度为：', n)

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
dhash1 = dhash(img1, (9, 8))
dhash2 = dhash(img2, (9, 8))
print('dhash1', dhash1)
print('dhash2', dhash2)
n = cmp_hash(dhash1, dhash2)
print('通过差值哈希计算两张图片的相似度为：', n)

