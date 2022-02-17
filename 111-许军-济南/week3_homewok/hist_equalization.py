# -- coding:utf-8 --
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
mpl.rcParams['font.sans-serif']=['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus']=False  # 解决保存图像是负号'-'显示为方块的问题
path = "./img/lenna.png"

# Histogram equalization
gray_level = [0 for i in range(256)]
def hist_equalizaation(img):
    #Histogram
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray_level[img[i][j]] += 1

    # gray scale
    gray_scale = [i/(img.shape[0]*img.shape[1]) for i in gray_level]

    # create target image
    tar_img = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixl = img[i][j]
            tar_img[i][j] = int(sum(gray_scale[:pixl+1])*256-1)

    # Create a histogram of the target image
    tar_level = [i for i in range(256)]
    for i in range(tar_img.shape[0]):
        for j in range(tar_img.shape[1]):
            tar_level[tar_img[i][j]] += 1
    return gray_level,tar_img,tar_level

if __name__ == '__main__':
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_level,img2,tar_level = hist_equalizaation(img)
    plt.figure()
    plt.subplot(221)
    plt.title("原图")
    plt.imshow(img,cmap="gray")

    plt.subplot(222)
    plt.title("原直方图")
    plt.bar([i for i in range(256)],gray_level)

    plt.subplot(223)
    plt.title("均衡化图")
    plt.imshow(img2,cmap="gray")

    plt.subplot(224)
    plt.title("均衡化直方图")
    plt.bar([i for i in range(256)],tar_level)
    plt.show()


