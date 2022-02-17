import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

img = cv2.imread("lenna.png", 0)  # 第二个参数： 1：彩色模式  0：灰度模式  -1：alpha模式

def grayequalization(img):
    a=[]
    for i in range(256):
        a.append(0)
    srch, srcw = img.shape
    for i in range(srch):
        for j in range(srcw):
            a[img[i,j]]+=1

   # 打印原始直方图
    plt.figure()
    mpl.rcParams['font.sans-serif'] = 'KaiTi'  # 字体类型参数找CSDN
    plt.title("原始直方图")
    plt.hist(img.ravel(),256)
    plt.show()

    # 进行变换
    cv2.imshow("oringinimg", img)
    t={}
    sum=0
    for i in range(256):
        if a[i]:
            sum=sum+a[i]
            t.setdefault(i,sum*256/(srcw*srch)-1)
    for i in range(srch):
        for j in range(srcw):
            img[i,j]=t[img[i,j]]
    cv2.imshow("grayequalization", img)
    cv2.waitKey(0)

    #均衡化后的直方图
    plt.figure()
    mpl.rcParams['font.sans-serif'] = 'KaiTi'  # 字体类型参数找CSDN
    plt.title("均衡直方图")
    plt.hist(img.ravel(), 256)
    plt.show()


grayequalization(img)