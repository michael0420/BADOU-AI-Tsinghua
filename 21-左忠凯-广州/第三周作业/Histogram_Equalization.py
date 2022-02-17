import cv2
import numpy as np
from matplotlib import pyplot as plt


def hist_qeual(img_gray):
    sumPi = 0
    h = img_gray.shape[0]
    w = img_gray.shape[1]
    sum_pix = h * w         #总像素个数

    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 255]) # 得到灰度图像原来的直方图
    new_hist = np.zeros([256], np.uint8)    # 均衡化以后的直方图
    new_img = np.zeros([h, w], img_gray.dtype)     # 均衡化以后的新图像

    for i in range(256):    # 遍历每个灰度值
        pix_num = hist[i]   # 每个灰度值对应的像素数量
        pi = float(pix_num) / sum_pix
        sumPi += pi         # 累加直方图
        value = sumPi * 256 - 1
        if value < 0:       # 处理均衡化后小于0
            value = 0
        new_hist[i] = round(value) # 四舍五入得到均衡化以后的直方图

    # 遍历所有像素点，修改像素点对应的灰度值
    for i in range(h):
        for j in range(w):
            value = img_gray[i, j]
            new_img[i, j] = new_hist[value]
    return new_img


# 1、得到灰度图片
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建画布
plt.figure(figsize=(6, 10), dpi=100)        # 画布10*10寸，dpi=100
plt.subplots_adjust(wspace=0.3, hspace=0.3) # 子图横竖间隔0.3英寸

# 2、显示灰度图片
plt.subplot(4, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("Origin gray img")

# 3、显示直方图
hist1 = cv2.calcHist([img_gray], [0], None, [256], [0, 255])
plt.subplot(4, 2, 2)
plt.plot(hist1)
plt.xlim(0, 255)
plt.title("Origin Gray Histogram")

# 4、显示直方图均衡化以后的图片
plt.subplot(4, 2, 3)
img_equal = hist_qeual(img_gray)
plt.imshow(img_equal, cmap='gray')
plt.title("My Equa img")

# 5、显示均衡化以后的直方图
hist2 = cv2.calcHist([img_equal], [0], None, [256], [0, 255])
plt.subplot(4, 2, 4)
plt.plot(hist2)
plt.xlim(0, 255)
plt.title("My Equali Histogram")

# 6、直接调用函数完成直方图均衡化
dst_img = cv2.equalizeHist(img_gray)
dst_hist = cv2.calcHist([dst_img], [0], None, [256], [0, 255])
plt.subplot(4, 2, 5)
plt.imshow(dst_img, cmap='gray')
plt.title("cv2 equali img")

plt.subplot(4, 2, 6)
plt.plot(dst_hist)
plt.xlim(0, 255)
plt.title("cv2 Equali Histogram")

# 7、彩色图像直方图均衡化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(4, 2, 7)
plt.imshow(img_gray)
plt.title("rgb img")

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
new_img = cv2.merge((bH, gH, rH))
new_img_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
plt.subplot(4, 2, 8)
plt.imshow(new_img_rgb)
plt.title("new rgb img")

plt.show()
cv2.waitKey(0)



