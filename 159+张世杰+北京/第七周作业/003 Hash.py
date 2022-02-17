# encoding: utf-8
import cv2
import numpy as np

'''均值hash'''
def ahash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)  # 采用双线性差值的方式缩小图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    '''求像素均值'''
    avg = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            avg += gray[i, j] / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

'''差值hash'''
def dhash(img):
    img = cv2.resize(img,(8,9), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for j in range(8):
        for i in range(8):
            if gray[i,j] > gray[i+1,j]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

img = cv2.imread('lenna.png', 1)
print('均值算法哈希：',ahash(img))
print('差值算法哈希：',dhash(img))
