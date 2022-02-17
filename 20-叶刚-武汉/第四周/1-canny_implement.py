"""
@GiffordY
Implementation of Canny edge detection algorithm
"""
import numpy as np
import cv2 as cv


# 根据高斯核尺寸和标准差，计算得到二维高斯核
def gen_gaussian_2d_kernel(ksize: int, sigma=1.0):
    # 参数检查
    if ksize % 2 == 0:
        ksize = ksize + 1
    if sigma <= 0 or sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    # 二维高斯核
    kernel_2d = np.zeros([ksize, ksize], np.float32)
    a = 1 / (2 * np.pi * sigma ** 2)
    b = -1 / (2 * sigma ** 2)
    k = int((ksize - 1) / 2)  # 核的中心，由ksize = 2k + 1计算得出
    for j in range(0, ksize):
        for i in range(0, ksize):
            kernel_2d[i, j] = a * np.exp(b * ((i-k)**2 + (j-k)**2))
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


# 通过调用OpenCV算子，计算得到的二维高斯核
def gen_gaussian_2d_kernel_cv(size, sigma=1.0):
    if size % 2 == 0:
        size = size + 1
    kernel_1d = cv.getGaussianKernel(size, sigma, ktype=cv.CV_32F)
    kernel_2d = np.dot(kernel_1d, kernel_1d.T)
    return kernel_2d


# 使用二维高斯核，进行高斯滤波
def gaussian_blur(src_img, ksize: int, sigma=1.0):
    height, width = src_img.shape[:2]
    assert height != 0 or width != 0
    if ksize % 2 == 0:
        ksize = ksize + 1
    kernel = gen_gaussian_2d_kernel(ksize, sigma)
    dst_img = np.zeros(src_img.shape, np.float32)
    # 图像边缘填充
    pad_img = np.pad(src_img, ((ksize//2, ksize//2), (ksize//2, ksize//2)), mode='constant')
    # 高斯滤波
    for row in range(src_img.shape[0]):
        for col in range(src_img.shape[1]):
            dst_img[row, col] = np.sum(pad_img[row:row+ksize, col:col+ksize] * kernel)
    #dst_img = dst_img.astype(dtype=np.uint8)
    return dst_img


# 检测图像中的水平、垂直和对角边缘（使用sobel矩阵求梯度，得到梯度幅值图像和梯度方向矩阵）
def calc_edges_by_sobel(src_img, dst_img=None, angle_matrix=None):
    # 参数初始化
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grad_x = np.zeros(src_img.shape)
    img_grad_y = np.zeros(src_img.shape)
    dst_img = np.zeros(src_img.shape)
    # sobel核是3x3的，这里给输入图像四周填充一圈像素0
    pad_img = np.pad(src_img, ((1, 1), (1, 1)), mode='constant', constant_values=((0, 0), (0, 0)))
    for row in range(src_img.shape[0]):
        for col in range(src_img.shape[1]):
            img_grad_x[row, col] = np.sum(pad_img[row:row+3, col:col+3] * sobel_kernel_x)   # x方向
            img_grad_y[row, col] = np.sum(pad_img[row:row+3, col:col+3] * sobel_kernel_y)   # y方向
            dst_img[row, col] = np.sqrt(img_grad_x[row, col]**2 + img_grad_y[row, col]**2)   # 梯度幅值
    img_grad_x[img_grad_x == 0] = 0.00000001    # 避免下一步的计算出现除数为0的情况
    angle_matrix = img_grad_y / img_grad_x      # 此处用tan(theta)的值表示方向，而不是theta
    return dst_img, angle_matrix


# 非极大值抑制
def non_maximum_suppression(img_grad, angle_matrix):
    img_sup = np.zeros(img_grad.shape)
    for row in range(1, img_grad.shape[0]-1):
        for col in range(1, img_grad.shape[1]-1):
            flag = True     # 标记在8邻域内是否要抹去
            tmp = img_grad[row-1:row+2, col-1:col+2]    # 梯度幅值的8邻域
            # 根据梯度的方向，使用线性插值法判断抑制与否
            if angle_matrix[row, col] <= -1:
                p1 = (tmp[0, 1] - tmp[0, 0]) / angle_matrix[row, col] + tmp[0, 1]
                p2 = (tmp[2, 1] - tmp[2, 2]) / angle_matrix[row, col] + tmp[2, 1]
            elif angle_matrix[row, col] >= 1:
                p1 = (tmp[0, 2] - tmp[0, 1]) / angle_matrix[row, col] + tmp[0, 1]
                p2 = (tmp[2, 0] - tmp[2, 1]) / angle_matrix[row, col] + tmp[2, 1]
            elif angle_matrix[row, col] > 0:
                p1 = (tmp[0, 2] - tmp[1, 2]) / angle_matrix[row, col] + tmp[1, 2]
                p2 = (tmp[2, 0] - tmp[1, 0]) / angle_matrix[row, col] + tmp[1, 0]
            elif angle_matrix[row, col] < 0:
                p1 = (tmp[1, 0] - tmp[0, 0]) / angle_matrix[row, col] + tmp[1, 0]
                p2 = (tmp[1, 2] - tmp[2, 2]) / angle_matrix[row, col] + tmp[1, 2]
            if not (img_grad[row, col] > p1 and img_grad[row, col] > p2):
                flag = False
            if flag:
                img_sup[row, col] = img_grad[row, col]
    return img_sup


# 双阈值（滞后阈值）检测和连接强、弱边缘，剔除假边缘
def hysteresis_threshold(img_sup, threshold1, threshold2, img_edge=None):
    assert threshold1 < threshold2
    img_edge = img_sup.copy()
    # 遍历所有像素点, 大于高阈值的为强边缘点，将强边缘点置255，同时保留其坐标（进栈），强边缘点直接就是真边缘点
    # 低于低阈值的为假边缘点，将假边缘点置0（舍弃）
    stack = []
    for i in range(1, img_edge.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_edge.shape[1] - 1):
            if img_edge[i, j] >= threshold2:  # 强边缘点，保留
                img_edge[i, j] = 255
                stack.append([i, j])
            elif img_edge[i, j] <= threshold1:  # 假边缘点，舍弃
                img_edge[i, j] = 0
    print('第一轮，真边缘点数量：', len(stack))
    # 大于低阈值且小于高阈值的像素点，为弱边缘点。弱边缘点需要进一步判断是否为真边缘点
    # 遍历所有真边缘点，查看其8邻域内是否存在弱边缘点，若存在弱边缘点，则认为该弱边缘点是真边缘点，进栈

    while not len(stack) == 0:
        y0, x0 = stack.pop()  # 真边缘点出栈
        a = img_edge[y0 - 1:y0 + 2, x0 - 1:x0 + 2]   # 8领域像素
        for i in range(y0-1, y0+2):
            for j in range(x0-1, x0+2):
                if i == y0 and j == x0:     # 中心点已经是真边缘点，不用再判断
                    continue
                if (img_edge[i, j] > threshold1) and (img_edge[i, j] < threshold2):
                    img_edge[i, j] = 255    # 这个弱边缘点标记为真边缘
                    stack.append([i, j])    # 进栈

    # 将所有非真实边缘点置0
    img_edge[np.where(img_edge != 255)] = 0
    return img_edge


# canny边缘检测算法实现
def canny_implement(src_img, threshold1, threshold2, edges=None):
    # 参数检查
    assert threshold1 < threshold2
    assert threshold1 >= 0 and threshold1 < 255
    assert threshold2 <= 255
    # 1、灰度化
    if len(src_img.shape) == 3 and src_img.shape[2] == 3:
        img_gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    else:
        img_gray = src_img
    # 2、高斯滤波
    img_gauss = gaussian_blur(img_gray, 3, 1)
    # 3、检测图像中的水平、垂直和对角边缘
    img_grad, angle_matrix = calc_edges_by_sobel(img_gauss)
    # 4、非极大值抑制，剔除非边缘点
    img_sup = non_maximum_suppression(img_grad, angle_matrix)
    # 5、双阈值（滞后阈值）检测和连接强、弱边缘，剔除假边缘
    img_edge = hysteresis_threshold(img_sup, threshold1, threshold2)
    return img_edge


if __name__ == '__main__':
    kernel_2d_cv = gen_gaussian_2d_kernel_cv(3, 1)
    print(kernel_2d_cv)
    print("==========================================")
    kernel_2d_my = gen_gaussian_2d_kernel(3, 1)
    print(kernel_2d_my)

    img_origin = cv.imread('lenna.png', 1)
    img_edges = canny_implement(img_origin, 50, 150)

    cv.imshow('img_origin', img_origin)
    cv.imshow('img_edges', img_edges)
    cv.waitKey(0)
