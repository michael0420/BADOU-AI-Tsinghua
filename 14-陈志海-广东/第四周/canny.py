import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

PI = math.pi


def getGaussKernel(ksize, sigma):
    """
    函数：用于生成奇数大小的高斯卷积核
    ksize: 高斯核大小=ksize * ksize
    sigma：高斯函数标准差
    """
    if ksize < 3:       # 高斯卷积核大小，必须为不小于3的奇数
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1

    value = [i - ksize//2 for i in range(ksize)]
    gauss_kernel = np.zeros((ksize, ksize))
    # n1 = 1 / (sigma * (2*PI)**0.5)
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma**2)
    for i in range(ksize):
        x = value[i]
        for j in range(ksize):
            y = value[j]
            gauss_kernel[i, j] = n1 * math.exp(n2 * (x**2 + y**2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()        # 归一化
    return gauss_kernel


def gaussFilter(img, gauss_kernel):
    """
    高斯滤波
    :param img: 原始图像，灰度图
    :param gauss_kernel: 高斯卷积核
    :return: 高斯卷积后得图像
    """
    k = gauss_kernel.shape[0]       # 卷积核大小
    p = k//2
    a = np.zeros((3,3))
    img_pad = np.pad(img, ((p, p), (p, p)), 'constant')
    img_gauss = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_gauss[i, j] = np.sum(img_pad[i:i+k, j:j+k] * gauss_kernel)
    return img_gauss


def sobelConvolution(img):
    """
    sobel卷积计算梯度
    :param img: 灰度图像
    :return: sobel卷积后得梯度图像，以及y方向与x方向梯度比值，即tan
    """
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    img_pad = np.pad(img, ((1, 1), (1, 1)), "constant")
    grad_x = np.zeros(img.shape)        # x方向梯度
    grad_y = np.zeros(img.shape)        # y方向梯度
    img_sobel = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_x)
            grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_y)
            img_sobel[i, j] = np.sqrt(grad_x[i, j]**2 + grad_y[i, j]**2)
    grad_x[grad_x == 0] = 0.00000001
    angle = grad_y / grad_x
    return img_sobel, angle


def doNMS(img, angle):
    """
    计算非极大值抑制
    :param img:灰度图像的梯度矩阵
    :param angle:梯度矩阵中每个点的y向梯度和x向梯度的tan值
    :return:非极大值抑制处理后得梯度矩阵
    """
    dx, dy = img.shape
    img_nms = np.zeros(img.shape)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            flag = True
            temp = img[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]

            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]


            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]

            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            else:
                img_nms[i, j] = img[i, j]
            if not (temp[1, 1] > num_1 and temp[1, 1] > num_2):
                flag = False
            if flag:
                img_nms[i, j] = img[i, j]
    return img_nms


def doDualThreshold(img, thresh_1, thresh_2):
    """
    双阈值计算
    :param img: NMS算法处理过的梯度矩阵
    :param thresh_1: 低阈值
    :param thresh_2: 高阈值
    :return: canny边缘检测结果
    """
    edgePoint = []       # 待判断边缘的点
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] >= thresh_2:       # 边缘=255
                img[i, j] = 255
                edgePoint.append([i, j])
            elif img[i, j] <= thresh_1:       # 非边缘=0
                img[i, j] = 0

    while not len(edgePoint) == 0:
        i, j = edgePoint.pop()
        if i != 0 and j != 0 and i != (img.shape[0]-1) and j != (img.shape[0]-1):
            a = img[i-1:i+2, j-1:j+2]
            if (a[0, 0] > thresh_1) and (a[0, 0] < thresh_2):
                img[i-1, j-1] = 255
                edgePoint.append([i-1, j-1])
            if (a[0, 1] > thresh_1) and (a[0, 1] < thresh_2):
                img[i-1, j] = 255
                edgePoint.append([i-1, j])
            if (a[0, 2] > thresh_1) and (a[0, 2] < thresh_2):
                img[i-1, j+1] = 255
                edgePoint.append([i-1, j+1])
            if (a[1, 0] > thresh_1) and (a[1, 0] < thresh_2):
                img[i, j-1] = 255
                edgePoint.append([i, j-1])
            if (a[1, 2] > thresh_1) and (a[1, 2] < thresh_2):
                img[i, j+1] = 255
                edgePoint.append([i, j+1])
            if (a[2, 0] > thresh_1) and (a[2, 0] < thresh_2):
                img[i+1, j-1] = 255
                edgePoint.append([i, j+1])
            if (a[2, 1] > thresh_1) and (a[2, 1] < thresh_2):
                img[i+1, j] = 255
                edgePoint.append([i, j+1])
            if (a[2, 2] > thresh_1) and (a[2, 2] < thresh_2):
                img[i+1, j+1] = 255
                edgePoint.append([i+1, j+1])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 255 and img[i, j] != 0:
                img[i, j] = 0
    return img


img = cv2.imread("lenna.png", 0)
kernel = getGaussKernel(ksize=10, sigma=1.52)
img_gauss = gaussFilter(img, kernel)
img_sobel, angle = sobelConvolution(img_gauss)
img_nms = doNMS(img_sobel, angle)
thresh_1 = img_sobel.mean() * 0.5
thresh_2 = thresh_1*3
img_canny = doDualThreshold(img_nms, thresh_1, thresh_2)

if_plot = True
if if_plot:
    plt.figure()
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1, rowspan=1)
    ax1.imshow(img, cmap="gray")
    ax1.set_title("img")
    ax1.set_axis_off()

    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1, rowspan=1)
    ax2.imshow(img_gauss, cmap="gray")
    ax2.set_title("img_gauss")
    ax2.set_axis_off()

    ax3 = plt.subplot2grid((2, 4), (1, 0), colspan=1, rowspan=1)
    ax3.imshow(img_sobel, cmap="gray")
    ax3.set_title("img_sobel")
    ax3.set_axis_off()

    ax4 = plt.subplot2grid((2, 4), (1, 1), colspan=1, rowspan=1)
    ax4.imshow(img_nms, cmap="gray")
    ax4.set_title("img_nms")
    ax4.set_axis_off()

    ax5 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
    ax5.imshow(img_canny, cmap="gray")
    ax5.set_title("img_canny")
    ax5.set_axis_off()

    plt.show()
