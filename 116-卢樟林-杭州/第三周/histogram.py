#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2021/12/12 12:03:55
@Author : LuZhanglin

直方图均衡化
'''

import cv2
import numpy as np

gray = cv2.imread("lenna.png", 0)
h, w = gray.shape
uni_val = np.unique(gray)
uni_val.sort()

val_cont = [(gray == val).sum() for val in uni_val]

val_eq = np.round(np.cumsum(val_cont) * 256 / (h*w) - 1).astype(int)

cvt_dic = {k:v for k, v in zip(uni_val, val_eq)}

def vec_translate(a, my_dict):
    # 将原图对应位置灰度替换为均衡化后的灰度
    return np.vectorize(my_dict.__getitem__)(a)

eq_gray = vec_translate(gray, cvt_dic)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].hist(gray.ravel(), label='original')
ax[1].hist(eq_gray.ravel(), label='equalization')
plt.show()



