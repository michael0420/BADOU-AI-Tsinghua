import numpy as np
import matplotlib.pyplot as plt
import  cv2

###########################################
#灰度图的直方图
#flag =True
flag =False
if flag:
    img = cv2.imread("lenna.png")
    # 将图像转化成灰度图像
    img_gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #参数1 ： 图像矩阵  eg  【image】
    #参数2 ： 图像的通道数
    #参数3 ： 掩膜，一般为None
    #参数4 ： 直方图大小，一般等于灰度级
    #参数5 ： x 轴范围
    hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    plt.figure() # 新建一个图像/画布
    plt.plot(hist,c="red")
    plt.title("grayscale Histogram") # 标题
    plt.xlabel("pixels")  # x轴标签
    plt.ylabel("num of pixels") # y轴标签
    plt.xlim([0,256])  # 设置x 坐标轴范围
    plt.show()

###############################################
# 三通道的直方图
flag =True
#flag =False
if flag:
    img = cv2.imread("lenna.png")
    # 将图像拆分成三个通道 bgr
    chans = cv2.split(img)

    colors =("b","g","r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for i in range(len(chans)):
        hist = cv2.calcHist([chans[i]],[0],None,[256],[0,256])
        plt.plot(hist,c=colors[i],label=colors[i])
        plt.legend(loc=0)
plt.show()