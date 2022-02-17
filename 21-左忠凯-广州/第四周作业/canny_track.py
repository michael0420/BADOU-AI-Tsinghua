import cv2
import numpy as np

lowThreshold = 0
max_lowThreshold = 100
HighThreshold = lowThreshold * 3
kernel_size = 3

# 读取图像
img = cv2.imread('lenna.png')
cv2.namedWindow("canny demo")

'''
高斯滤波函数原型
GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)，函数参数：
    src：需要滤波图像
    ksize：滤波器大小，ksize.width和ksize_height可以不同，但是比如为正数和奇数，也可以是0，然后根据sigma计算得出
    sigmaX：X方向上高斯核标准偏差
    dst：滤波后的图像
    sigmaY：Y方向上的高斯核心标准差
    borderType:判断图像边界的模式，默认为BORDER_DEFAULT
'''
def CannyThreshold(img, lowThre, HighThre, KSize):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
    gauss_edges = cv2.GaussianBlur(img_gray, (3, 3), 0) # 高斯核3x3，标准差为0
    detected_edges = cv2.Canny(gauss_edges, lowThre, HighThre, KSize)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    return dst

def updateImg(x):
    global lowThreshold, HighThreshold

    # 获取到滑动条的值
    lowThreshold = cv2.getTrackbarPos('Min Threshold', 'canny demo')
    HighThreshold = lowThreshold * 3

    dst_img = CannyThreshold(img, lowThreshold, HighThreshold, kernel_size)
    cv2.imshow('canny demo', dst_img)


# 设置调节杠杆
cv2.createTrackbar('Min Threshold', 'canny demo', lowThreshold, max_lowThreshold, updateImg)
updateImg(0) # 初始化

if cv2.waitKey(0) == 27: # 按下ESC键
    cv2.destroyAllWindows()






