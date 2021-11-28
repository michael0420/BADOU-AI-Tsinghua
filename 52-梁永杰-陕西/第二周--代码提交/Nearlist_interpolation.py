'''
@author:Jelly

将图像进行上采样放大处理，采用最邻近插值方法

相关接口：

def Near_Interp(img):
函数用途：对图像进行最邻近插值

'''

import matplotlib.pyplot as plt
import numpy as np

def Near_Interp(img,hight=800,width=800):
    '''
    使用最邻近插值的方式将图像进行放大
    :param img: 原始图像
    :param hight: 放大图像的高度
    :param width: 放大图像的宽度
    :return: 插值后的图像(没达到要求的参数，将返回原图片)
    '''
    h,w,channels = img.shape
    if hight<h or width<w:
        print('图片输入放大尺寸错误，当前图片尺寸为高=%d，宽=%d',h,w)
        return img
    Img = np.zeros((hight,width,channels),img.dtype)
    sh = hight/h
    sw = width/w
    for i in range(hight):
        for j in range(width):
            x = int(i/sh)
            y = int(j/sw)
            Img[i,j] = img[x,y]
    return Img

plt.subplot(221)
print("1.输出原图")
img = plt.imread('lenna.png')
plt.imshow(img)
plt.subplot(222)

Img = Near_Interp(img,800,800)
print("2.输出800*800像素图")
plt.imshow(Img)

plt.subplot(223)
Img2 = Near_Interp(img,1000,600)
print("3.输出1000*600像素图")
plt.imshow(Img2)
plt.show()