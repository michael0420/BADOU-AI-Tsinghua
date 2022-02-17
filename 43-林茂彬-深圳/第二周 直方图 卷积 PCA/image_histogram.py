import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
直方图是对图像像素的统计分布，它统计了每个像素（0到L-1）的数量。
直方图均衡化就是将原始的直方图拉伸，使之均匀分布在全部灰度范围内，从而增强图像的对比度。
直方图均衡化的中心思想是把原始图像的的灰度直方图从比较集中的某个区域变成在全部灰度范围内的均匀分布
'''

'''
calcHist—计算图像直方图
函数原型：cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) 
一、images（输入图像）参数必须用方括号括起来。
二、计算直方图的通道。
三、Mask（掩膜），一般用None，表示处理整幅图像。
四、histSize，表示这个直方图分成多少份（即多少个直方柱）。
五、range，直方图中各个像素的值，[0.0, 256.0]表示直方图能表示像素值从0.0到256的像素。
六、最后是两个可选参数，由于直方图作为函数结果返回了，所以第六个hist就没有意义了（待确定） 最后一个accumulate是一个布尔值，用来表示直方图是否叠加。
原文链接：https://blog.csdn.net/qq_42250840/article/details/104878333
'''


# 灰度图像的直方图

def img_histogram(img):
    img_source=cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    hist=cv2.calcHist([img_gray],[0],None,[256],[0,256],)
    plt.figure()  # 新建一个图像
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签
    plt.plot(hist)
    plt.xlim([0, 256])  # 设置x坐标轴范围
    plt.show()

# 灰度图像的直方图 第二种

def img_histogram2(img):
    img_source=cv2.imread("lenna.png")
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签
    plt.hist(img_source.ravel(), 256)
    plt.show()

# 彩色图像的直方图
def img_histogram3(img):
    image = cv2.imread("lenna.png")
    cv2.imshow("Original",image)
    #cv2.waitKey(0)

    chans=cv2.split(image)
    print(chans)
    colors = ("b","g","r")
    print(colors)
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan,color) in zip(chans,colors):
        hist = cv2.calcHist([chan],[0],None,[256],[0,256])
        plt.plot(hist,color = color)
        plt.xlim([0,256])
    plt.show()



if __name__ == '__main__':
    img_source = cv2.imread('lenna.png')
    #img_histogram = img_histogram(img_source)
    #img_histogram2 = img_histogram2(img_source)
    img_histogram3 = img_histogram3(img_source)

