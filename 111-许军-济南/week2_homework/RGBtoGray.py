# -- coding:utf-8 --
# RGB图转为灰度图常用的方法：加权求和，平均值
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus']=False  # 解决保存图像是负号'-'显示为方块的问题
path = "./img/lenna.png"
# 每个通道加权
def RGB_TO_Gray1(img):
    img1 = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray = 0.3*img[i][j][0] + 0.6*img[i][j][1] + 0.1*img[i][j][2]
            img1[i][j] = gray
    return img1
# 三个通道平均
def RGB_TO_Gray2(img):
    img2 = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray = (img[i][j][0] + img[i][j][1] + img[i][j][2])/3
            img2[i][j] = gray
    return img2
# opecv库转换
def Opencv_convert(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

# 转为二值图
def TO_Binary(path,threshold):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= threshold:
                img[i][j] = 1
            else:
                img[i][j] = 0
    return  img
if __name__ == '__main__':
    img= cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img1 = RGB_TO_Gray1(img)
    img2 = RGB_TO_Gray2(img)
    img3 = Opencv_convert(path)
    img4 = TO_Binary(path,128)
    # 显示
    plt.figure()
    plt.subplot(2,2,1)
    plt.title("原图")
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.title("加权平均灰度图")
    plt.imshow(img1,cmap="gray")
    plt.subplot(2,2,3)
    plt.imshow(img2,cmap="gray")
    plt.title("通道平均灰度")
    plt.subplot(2,2,4)
    plt.imshow(img3,cmap="gray")
    plt.title("opencv处理的灰度图")
    # 二值图像显示
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("原图")
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title("二值化图")
    plt.imshow(img4,cmap="gray")
    plt.show()

