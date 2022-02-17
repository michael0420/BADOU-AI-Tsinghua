# encoding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    '''1  # 灰度化'''
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型，
    img = img.mean(axis=-1)  # 取均值就是灰度化了，此时的图片不能展示，取均值后还是浮点数：axis = -1 把各通道的值换成均值；
    '''2  # 高斯平滑：通过设定sigma 的值来确定高斯卷积核的大小；'''
    sigma = 0.5  # 高斯函数的标准差值， 由此来直接确定卷积核的值，同时确定卷积核的大小；
    dim = int(np.round(6 * sigma + 1))  # 3*sigma 包含了99.7的信息，所以卷积核的直径应该为6*sigma；是否+1 不清楚影响
    if dim % 2 == 0:  # 最好是奇数,不是的话加一，卷积核应为奇数；
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了，创建一个0值矩阵；
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列，dim//2 求得卷积核半径，生成的序列作为卷积核的横纵坐标
    '''计算高斯核：直接套用公式'''
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()  # 归一化， 让卷积核的值和为1，保证亮度不变；
    '''高斯卷积'''
    dx, dy = img.shape  # 求得img的宽核高；
    img_new = np.zeros(img.shape)  # 默认数据类型为浮点型；
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 填充：横轴和纵轴都填充tmp像素， 填充值为0
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)  # 卷积
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    '''3# Soble：边缘检测'''
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    '''计算梯度：图像处理重的梯度， 并不是标准的数学函数梯度'''
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # 获得x方向的梯度
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # 获得y方向的梯度
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 获得梯度值
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 0值不能作为被除数，
    angle = img_tidu_y / img_tidu_x  # 注意此处的出发， 计算方式和上面的乘法一致，确定梯度的方向，角度的正切值；
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    '''4# 非极大值抑制:此处是根据梯度值的大小来决定是否抑制:此处的过程比较了解， 但是比较蒙'''
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵

            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    '''5# 双阈值检测&边缘连接'''
    '''舍弃低阈值下，保留高阈值，同时将高一阈值的像素点位置保存在列表里面'''
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    '''找出以高阈值的点为中心的3✖️3个点，分别和高阈值进行比较， 决定是否保留'''
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()
        while not len(zhan) == 0:
            temp_1, temp_2 = zhan.pop()  # 出栈
            a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
                zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2] = 255
                zhan.append([temp_1 - 1, temp_2])
            if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
                img_yizhi[temp_1 - 1, temp_2 + 1] = 255
                zhan.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
                img_yizhi[temp_1, temp_2 - 1] = 255
                zhan.append([temp_1, temp_2 - 1])
            if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
                img_yizhi[temp_1, temp_2 + 1] = 255
                zhan.append([temp_1, temp_2 + 1])
            if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
                img_yizhi[temp_1 + 1, temp_2 - 1] = 255
                zhan.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
                img_yizhi[temp_1 + 1, temp_2] = 255
                zhan.append([temp_1 + 1, temp_2])
            if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
                img_yizhi[temp_1 + 1, temp_2 + 1] = 255
                zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()