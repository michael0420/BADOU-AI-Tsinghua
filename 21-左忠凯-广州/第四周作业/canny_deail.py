import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ == '__main__':
    # 1、读取图片
    img = plt.imread('lenna.png')
    img = img * 255 # 这里存储的png图片数据采用0~1的浮点数据，因此要乘以255，得到0~255数据

    # 2、灰度化
    img_gray = img.mean(axis=-1)  # 取均值，进行灰度化

    # 3、高斯平滑
    sigma = 0.5 # 高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # 根据标准差计算出高斯核是N*N矩阵，一般核宽度半径与σ的比值差不多为3倍就可以，包含了99%的信息，
                                        # 这里乘以6是计算的直径，也就是2*3=6
    if dim % 2 == 0: # 高斯核大小要为奇数，
        dim += 1

    # 计算高斯核
    Gaussian_filter = np.zeros([dim, dim]) # 高斯滤波核
    tmp = [i-dim//2 for i in range(dim)] # 生成序列
    n1 = 1/(2 * math.pi * sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i][j] = n1*math.exp((n2 * (tmp[i]**2 + tmp[j]**2)))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

    dx = img_gray.shape[0]
    dy = img_gray.shape[1]
    img_gauss = np.zeros(img_gray.shape)
    tmp = dim//2  # 整除，用于pading，计算添加几圈
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant') # 对灰度图进行边缘填充

    # 进行高斯滤波
    for i in range(dx):
        for j in range(dy):
            img_gauss[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gaussian_filter) # 开始进行高斯滤波

    # 4、使用sobel算子检测边缘
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_gauss.shape)  # 保存梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_gauss.shape)
    img_pad = np.pad(img_gauss, ((1, 1), (1, 1)), 'constant') # padding

    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)   # 沿着X方向的偏导数
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)   # 沿着Y方向的偏导数
            img_tidu[i, j] = np.sqrt((img_tidu_x[i, j]**2) + img_tidu_y[i, j]**2)   # 根号下X^2+Y^2，得到梯度
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y/img_tidu_x                                                   # y/x得到梯度的角度 tan


    # 5、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True     # 标记是否要抹去
            # 使用线性插值法计算是否抑制
            temp = img_tidu[i-1:i+2, j-1:j+2] # 获得8邻域矩阵，取掉周围一圈，所以for循环是从1开始的，而不是0
            # 非极大值抑制，有四种图形，参考
            # https://blog.csdn.net/fengye2two/article/details/79190759
            # https://blog.csdn.net/u010551600/article/details/80507271
			# https://blog.csdn.net/kezunhai/article/details/11620357
            # 四个角度：0，45，90，135
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False    # 剔除掉
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False  # 剔除掉
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False  # 剔除掉
            elif angle[i, j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) / angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) / angle[i, j] + temp[1, 2]
                    if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                        flag = False  # 剔除掉
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]

    # 6、双阈值检测
    #lower_boundary = img_tidu.mean() * 0.5 # 去均值的0.5倍
    #high_boundary = lower_boundary * 3 # 高阈值是第阈值的3倍
    lower_boundary = 50 # 去均值的0.5倍
    high_boundary = 100 # 高阈值是第阈值的3倍
    img_dst = np.zeros(img_tidu.shape)

    # 先处理大于高阈值和低于低阈值的，高于高阈值的直接赋值255，低于低阈值的直接赋值0
    for i in range(1, img_yizhi.shape[0] - 1): # 最外圈不考虑
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:            # 强边缘
                img_dst[i, j] = 255
            elif img_yizhi[i, j] <= lower_boundary:         # 不是边缘
                img_dst[i, j] = 0
            else:                                           # 剩下的都是弱边缘的，先直接保存
                img_dst[i, j] = img_yizhi[i, j]

    # 处理，一定要先把强边缘以及非边缘标记出来以后才能处理弱边缘
    for i in range(1, img_dst.shape[0] - 1): # 最外圈不考虑
        for j in range(1, img_dst.shape[1] - 1):
            if (img_dst[i, j] != 255 and img_dst[i, j] != 0): # 既不是强边缘，也不属于非边缘，所以只有可能是弱边缘
                a = img_dst[i - 1: i + 2, j - 1: j + 2]  # 获得8邻域，判断8邻域里面是否有强边缘
                strong_edg_flag = False                 # 标记是否为强边缘，默认不是
                for x in range(3):
                    for y in range(3):
                        if a[x, y] == 255:              # 找到一个强边缘
                            strong_edg_flag = True      # 标记为强边缘
                            break                       # 只要找到一个强边缘就跳出
                if strong_edg_flag == True:
                    img_dst[i, j] = 255
                else:
                    img_dst[i, j] = 0


    # 显示图片
    plt.figure(figsize=(6, 8), dpi=100)  # 画布10*10寸，dpi=100
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 子图横竖间隔0.3英寸

    # 显示原始灰度图
    plt.subplot(3, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Origin gray img")

    # 显示高斯滤波后的图
    plt.subplot(3, 2, 2)
    plt.imshow(img_gauss.astype(np.uint8), cmap='gray') #高斯滤波后的图像是0~255的浮点数据
    plt.title("Gauss img")

    # 显示sobel边缘检测结果
    plt.subplot(3, 2, 3)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.title("Gauss img")

    # 显示非极大值抑制后的结果
    plt.subplot(3, 2, 4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.title("Suppre img")

    # 显示双阈值后的结果
    plt.subplot(3, 2, 5)
    plt.imshow(img_dst.astype(np.uint8), cmap='gray')
    plt.title("dual_thre img")


    plt.show()
    cv2.waitKey(0)


