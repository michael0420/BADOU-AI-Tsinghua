import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread("lenna.png")

'''
Sobel函数原型cv2.Sobel(src, ddepth, dx, dy[dst[,ksize[,scale[,delta[,borderType]]]]])
    src：原图像
    ddepth：输出图像深度，也就是数据类型，-1表示与原图像相同深度。可选的数据类型：CV_8U、CV_8S、CV_16S、CV_16U等等，比如CV_8U为8位无符号
            类型，范围是0~255，超出的范围会被截断，大于255的保存为255，小于0的保存为0
    dx，dy：dx=1，dy=0表示求X方向的一阶导数，当dx=0,dy=1的时候表示求Y方向的一阶导数。
    dst:目标图像
    ksize：Sobel算子大小，也就是卷积核大小，必须是1,3,5,7，默认是3。
    scale：计算结果放大比例，效果就是图更亮，默认为1
    delta：增量值，结果会加到最终的dst里面
    borderType：图像边界的模式，图像进行滤波操作的时候边缘的处理方式，默认为BORDER_DEFAULT
'''
# sobel求完导数以后结果会有负值，或大于255，因此要采用CV_16S
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)


# 将处理完后的数据使用函数convertScaleAbs转换为uint8格式
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

'''
将前面计算处理的独立的X,Y方向进行组合，使用函数addWeighted。此函数用于融合两幅图形函数原型：
addWeighted(src1, alpha, src2, beta, gamma, dst, dtype)
    src1：第一张图像
    alpha：第一个图像的权重，也就是alpha值
    src2：第二张图像
    beta：第二张图像的权重，值为1-alpha
    gamma：加到总和上的权重
    dst：输出的图像数组
    dtype：数据类型，默认设置为-1，也就是输出图像和原图像数据类型一直
'''
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

plt.figure(figsize=(6, 6), dpi=100)        # 画布10*10寸，dpi=100
plt.subplots_adjust(wspace=0.3, hspace=0.3) # 子图横竖间隔0.3英寸

# 显示原图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Origin rgb img")

# 显示X方向的边缘提取结果
img_absX = cv2.cvtColor(absX, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 2)
plt.imshow(img_absX)
plt.title("X aixs Sobel")

# 显示Y方向的边缘提取结果
img_absY = cv2.cvtColor(absY, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 3)
plt.imshow(img_absY)
plt.title("Y aixs Sobel")

# 显示X,Y叠加后的边缘提取结果
img_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 4)
plt.imshow(img_dst)
plt.title("dst aixs Sobel")

plt.show()


#显示图片
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
cv2.imshow("dst", dst)

cv2.waitKey(0)








