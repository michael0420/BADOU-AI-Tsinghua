#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2021/12/11 20:28:59
@Author : LuZhanglin

第二周作业
（1） 灰度图
（2） 二值图
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from pathlib import Path

def fig_path():
    """图片文件路径"""
    # 仓库路径包含中文，cv2读取有误
    # return str(Path(__file__).absolute().parent / 'lenna.png')
    return 'lenna.png'


def gray_image(img, mode):
    '''BRG转Gray
    1. 浮点算法 Gray = R0.3 + G0.59 + B0.11
    2. 整数方法 Gray = (R30 + G59 + B11)/100
    3. 移位方法 Gray = (R76 + G151 + B28)>>8
    4. 平均值法 Gray = (R+G+B) / 3
    5. 仅取绿色 Gray = G
    
    parameters
    -----------
    img : numpy.ndarray
        shape (h, w, channels)

    mode : int
        方法编号

    return
    -------
    np.ndarray, gray image
    '''
    return eval(f"gray_cvt{mode}")(img)


def gray_cvt1(img):
    """1. 浮点算法 Gray = R0.3 + G0.59 + B0.11"""
    return img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] * 0.3

def gray_cvt2(img):
    """2. 整数方法 Gray = (R30 + G59 + B11)/100"""
    return (img[:, :, 0] * 11 + img[:, :, 1] * 59 + img[:, :, 2] * 30) / 100

def gray_cvt3(img):
    """3. 移位算法 """
    return (img[:, :, 0] * 28 + img[:, :, 1] * 151 + img[:, :, 2] * 76) >> 8

def gray_cvt4(img):
    """4. 平均法"""
    return (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3

def gray_cvt5(img):
    """5. 仅取绿色 Gray = G"""
    return img[:, :, 1]


# BGR mode
img = cv2.imread(fig_path())
# numpy计算要先转化为int，不然在其中一些方法计算的时候结果会不正确
img_gray = gray_image(img.astype(int), mode=5)
img_gray = img_gray.astype(np.uint8)
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_gray, cmap='gray')
ax[1].imshow(img_gray2, cmap='gray')
plt.show()


#%%  二值化


img_binary = np.where(img_gray >= 130, 1, 0) 
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223) 
plt.imshow(img_binary, cmap='gray')
plt.show()




# %%
