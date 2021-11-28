# -- coding:utf-8 --
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus']=False  # 解决保存图像是负号'-'显示为方块的问题
path = "./img/lenna.png"
# 最近邻插值
def Nearset_Interpolation(img,size_x,size_y):
    height,width,channel = img.shape
    img0 = np.zeros((size_x,size_y,3),np.uint8)
    #print(type(img0[1][1][0]))
    scal_x = size_x/height
    scal_y = size_y/width
    for i in range(size_x):
        for j in range(size_y):
            x = int(i/scal_x)
            y = int(j/scal_y)
            img0[i,j] = img[x,y]
    return img0
'''
双线性插值相当于三次单线性插值，通过和相邻四个点的距离来表示每个点的权重
问题1：以左上角为坐标原点的话两边的像素可能会直接复制到目标图中，这样目标图像边缘的点没有考虑到四个点
问题2：如果不进行中心点对称，像素点偏向右下方，插值后的图像两个像素产生的边缘和原图像产生的边缘会有差异
'''
def Bilinear_Interpolation(img,size_x,size_y):
    height,width,channel = img.shape
    img0 = np.zeros((size_x,size_y,channel),np.uint8)
    scal_x,scal_y = size_x/height,size_y/width
    for k in range(3):
        for i in range(size_x):
            for j in range(size_y):
                x = (i + 0.5) / scal_x - 0.5
                y = (j + 0.5) / scal_y - 0.5
                src_x0 = int(x)
                src_y0 = int(y)
                src_x1 = min(src_x0 + 1,width - 1)
                src_y1 = min(src_y0 + 1,height - 1)
                temp0 = (src_y1-y) * img[src_x0,src_y0,k] + (y - src_y0) * img[src_x1,src_y0,k]
                temp1 = (src_y1-y) * img[src_x0,src_y1,k] + (y - src_y0) * img[src_x1,src_y0,k]
                img0[i,j,k] = int((src_x1 - x) * temp0 + (x - src_x0) * temp1)
    return img0


if __name__ == '__main__':
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img1 = Nearset_Interpolation(img,800,800)
    img2 = Bilinear_Interpolation(img,800,800)
    plt.figure()
    plt.subplot(2,2,1)
    plt.title("原图")
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.title("最近邻插值")
    plt.imshow(img1)
    plt.subplot(2,2,3)
    plt.title("双线性插值")
    plt.imshow(img2)
    plt.show()
