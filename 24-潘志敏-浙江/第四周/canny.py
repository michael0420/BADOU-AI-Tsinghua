import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #1.高斯平滑
    sigma = 0.5
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1

    Gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    #print(dim, tmp)
    n1 = 1 / (2 * math.pi * sigma**2)
    n2 = -1 / (2 * sigma**2)

    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i]**2 + tmp[j]**2))

    #print(Gaussian_filter)
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    #print(Gaussian_filter)

    dx, dy = img_gray.shape
    img_new = np.zeros(img_gray.shape)
    tmp = dim//2
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant')

    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gaussian_filter)

    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    #plt.show()

    #2.求梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grad_x = np.zeros(img_gray.shape)
    img_grad_y = np.zeros(img_gray.shape)
    img_grad = np.zeros(img_gray.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')

    for i in range(dx):
        for j in range(dy):
            img_grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x)
            img_grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y)
            img_grad[i, j] = np.sqrt(img_grad_x[i, j] ** 2 + img_grad_y[i, j] ** 2)

    img_grad_x[img_grad_x == 0] = 0.00000001
    angle = img_grad_y / img_grad_x

    plt.figure(2)
    plt.imshow(img_grad.astype(np.uint8), cmap='gray')
    plt.axis('off')
    #plt.show()

    #3.非极大值抑制
    img_nms = np.zeros(img_gray.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_grad[i-1:i+2, j-1:j+2]
            if angle[i, j] <= -1:#梯度更接近Y方向
                w = -1 / angle[i, j]
                num1 = w * temp[0, 0] + (1 - w) * temp[0, 1]
                num2 = w * temp[2, 2] + (1 - w) * temp[2, 1]
            elif angle[i, j] >= 1:
                w = 1 / angle[i, j]
                num1 = w * temp[0, 2] + (1 - w) * temp[0, 1]
                num2 = w * temp[2, 0] + (1 - w) * temp[2, 1]
            elif angle[i, j] > 0:
                w = angle[i, j]
                num1 = w * temp[2, 0] + (1 - w) * temp[1, 0]
                num2 = w * temp[0, 2] + (1 - w) * temp[1, 2]
            elif angle[i, j] < 0:
                w = -angle[i, j]
                num1 = w * temp[0, 0] + (1 - w) * temp[1, 0]
                num2 = w * temp[2, 2] + (1 - w) * temp[1, 2]

            if not (img_grad[i, j] > num1 and img_grad[i, j] > num2):
                flag = False

            if flag:
                img_nms[i, j] = img_grad[i, j]

    plt.figure(3)
    plt.imshow(img_nms.astype(np.uint8), cmap='gray')
    plt.axis('off')
    #plt.show()

    #4. 双阈值检测，连接边缘。遍历强边缘点，查看8领域是否存在有可能是边缘的点
    lower_T = img_grad.mean() * 0.5
    high_T = lower_T * 3
    zhan = []

    for i in range(1, img_nms.shape[0] - 1):
        for j in range(1, img_nms.shape[1] - 1):
            if img_nms[i, j] >= high_T: #强边缘点
                img_nms[i, j] = 255
                zhan.append([i, j])
            elif img_nms[i, j] <= lower_T:
                img_nms[i, j] = 0

    while not len(zhan) == 0:
        m, n = zhan.pop()
        area8 = img_nms[m-1:m+2, n-1:n+2]
        for i in range(0, 2):
            for j in range(0, 2):
                if area8[i, j] < high_T and area8[i, j] > lower_T:
                    img_nms[m-1+i, n-1+j] = 255
                    zhan.append([m-1+i, n-1+j])

    for i in range(img_nms.shape[0]):
        for j in range(img_nms.shape[1]):
            if img_nms[i, j] != 0 and img_nms[i, j] != 255:
                img_nms[i, j] = 0

    plt.figure(4)
    plt.imshow(img_nms.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()