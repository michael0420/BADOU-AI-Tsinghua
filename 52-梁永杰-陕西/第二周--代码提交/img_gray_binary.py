'''
@author:Jelly

彩色图像的灰度化与二值化

相关接口：
def ImgToGray_Self(img):
函数用途：彩色图像的灰度化

def GrayToBinary_Self(img_gray):
函数用途：灰度图像二值化


def ImgToBinary_Self(img):
函数用途：彩色图像的二值化

'''


import numpy as np
import matplotlib.pyplot as plt
import cv2

####################################################################

def ImgToGray_Self(img):
    '''
    使用matplotlib读取的图片数据为小数，opencv读入是0-255的色阈值
    :param img:读如的图片数据（可以是plt读如的数据，也可以是opencv读入的数据）
    :return:正常的色阈值为0-255的灰度图片数据（输出时应该使用cmap='gray'）
    '''
    h, w = img.shape[:2]
    img_gray = np.zeros([h,w],img.dtype)
    if isinstance(img[0,0,0],np.float32):#计算输入是plt读入的数据
        for i in range(h):
            for j in range(w):
                k = img[i, j]
                img_gray[i, j] = int(k[0] * 30 + k[1] * 59 + k[2] * 11)
    else: #计算输入的是cv读入的数据
        for i in range(h):
            for j in range(w):
                k = img[i, j]
                img_gray[i, j] = int(k[0] * 0.11 + k[1] * 0.59 + k[2] * 0.30)
    return img_gray



img_plt =plt.imread('lenna.png')   #RGB显示 float32
img_cv = cv2.imread('lenna.png') #BGR显示 0-255 int

img_gray_plt = ImgToGray_Self(img_plt)
img_gray_cv = ImgToGray_Self(img_cv)

print("1.用plt库读取图片")
plt.imshow(img_gray_plt,cmap='gray')
plt.show()

print("2.用cv库读取图片")
plt.imshow(img_gray_cv,cmap='gray')
plt.show()

#######################################################

def GrayToBinary_Self(img_gray):
    '''
    将彩色图片转灰度图片接口函数返回结果灰度图输入到该接口中，返回函数即为转换出来的黑白二值化图片
    :param img_gray: 灰度图片
    :return: 黑白二值化图片
    '''
    h,w = img_gray.shape  #灰度图只有一维数据
    for i in range(h):
        for j in range(w):
            if(float(img_gray[i,j])/255>=0.5):
                img_gray[i,j] = 255
            else:
                img_gray[i,j] = 0
    return img_gray

#利用上面转换的灰度图数据 img_gray_cv
img_binary = GrayToBinary_Self(img_gray_cv)

print("3.灰度图片转换为黑白图")
plt.imshow(img_binary,cmap='gray')
plt.show()


########################################################

def ImgToBinary_Self(img):
    '''
    读入彩色图片转化为黑白图片
    :param img: 彩色图片
    :return: 黑白图片
    '''
    img_gray = ImgToGray_Self(img)
    img_binary = GrayToBinary_Self(img_gray)
    return img_binary

img = cv2.imread('lenna.png')
img_binary = ImgToBinary_Self(img)

print("4.彩色图片直接转换为黑白图")
plt.imshow(img_binary,cmap='gray')
plt.show()

###########################################################

from skimage.color import rgb2gray

print("5.一张上面有两副图的，使用skimge库方法进行灰度转换")
#使用skmage库进行转换 转换图片为3分图片
img = cv2.imread('lenna.png')
img_gray = rgb2gray(img)

plt.subplot(121)
plt.imshow(img_gray,cmap='gray')


img_binary = np.where(img_gray > 0.5,1,0)

plt.subplot(122)
plt.imshow(img_binary,cmap='gray')
plt.show()

