import cv2
import numpy as np
from matplotlib import pyplot as plt


def my_hist(img):
    h = img.shape[0]
    w = img.shape[1]
    img_hist = np.zeros([256], np.uint32) #定义一个包含256个元素的数组

    # 遍历整副图像，统计灰度信息，也就是直方图信息
    for i in range(h):
        for j in range(w):
            value = img[i][j]
            img_hist[value] += 1
    return img_hist


# 1、得到灰度图片
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#创建画布
plt.figure(figsize=(6, 10), dpi=100) #画布10*10寸，dpi=100
plt.subplots_adjust(wspace=0.3, hspace=0.3) #子图横竖间隔0.3英寸
# 2、显示灰度图片
plt.subplot(3, 2, 1)
plt.imshow(img_gray, cmap='gray')

# 3、显示直方图
plt.subplot(3, 2, 2)
plt.hist(img_gray.ravel(), 256)

# 4、第二种，调用自己的函数，获取直方图
plt.subplot(3, 2, 3)
hist = my_hist(img_gray)
plt.plot(hist)
plt.xlim(0, 255)
plt.title("My Hist")

# 5、第三种方法绘制直方图
hist1 = cv2.calcHist([img_gray], [0], None, [256], [0, 255])
plt.subplot(3, 2, 4)
plt.plot(hist1)
plt.xlim(0, 255)
plt.title("Grayscale Histogram")

# 6、彩色直方图绘制
plt.subplot(3, 2, 5)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Original image")

plt.subplot(3, 2, 6)
plt.title("Color Histogram")
chans = cv2.split(img)  # 分离出图像的R、G、B三个通道的颜色
colors = ('b', 'g', 'r')

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 255])
    plt.plot(hist, color=color)
    plt.xlim(0, 255)

plt.show()
