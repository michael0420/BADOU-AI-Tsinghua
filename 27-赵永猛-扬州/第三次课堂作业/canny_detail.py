import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)#
    sigma = 0.5
    dim = int(np.round(6*sigma+1))
    if dim % 2 == 0:
        dim = dim + 1
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -2*sigma**2
    Gaussi_filter = np.zeros([dim, dim])
    tmp = [i- dim//2 for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            Gaussi_filter[i, j] = n1 * math.exp(n2 * (tmp[i]**2 + tmp[j]**2))
    Gaussi_filter = Gaussi_filter/Gaussi_filter.sum()
    print("高斯核：", Gaussi_filter)
    new_img = np.zeros(img.shape)
    dx, dy = img.shape
    img_pad = np.pad(img, ((dim//2, dim//2), (dim//2, dim//2)), 'constant')
    for i in range(dx):
        for j in range(dy):
            new_img[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gaussi_filter)
    plt.figure(1)
    plt.imshow(new_img.astype(np.uint8), cmap='gray')
    plt.axis('off')
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    gradient_x = np.zeros([dx, dy])
    gradient_y = np.zeros([dx, dy])
    gradient = np.zeros([dx, dy])
    yizhi_img = np.zeros([dx, dy])
    for i in range(dx):
        for j in range(dy):
            gradient_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_x_kernel)
            gradient_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_y_kernel)
            gradient[i, j] = np.sqrt(gradient_x[i, j]**2+gradient_y[i, j]**2)
    gradient_x[gradient_x == 0] = 0.00000001
    angle = gradient_y / gradient_x
    plt.figure(2)
    plt.imshow(gradient.astype(np.uint8), cmap='gray')
    plt.axis('off')

    for i in range(1, dx-1):
        for j in range(1, dy-1):
            inter_win = gradient[i-1:i+2, j-1:j+2]
            flag = False
            if angle[i, j] <= -1:
                a_ori = (inter_win[0, 1] - inter_win[0, 0])/angle[i, j] + inter_win[0, 1]
                b_ori = (inter_win[2, 1] - inter_win[2, 2])/angle[i, j] + inter_win[2, 1]
                if gradient[i, j] > a_ori and gradient[i, j] > b_ori:
                    flag = True

            elif angle[i, j] >= 1:
                a_ori = (inter_win[0, 2] - inter_win[0, 1])/angle[i, j] + inter_win[0, 2]
                b_ori = (inter_win[2, 0] - inter_win[2, 1])/angle[i, j] + inter_win[2, 0]
                if gradient[i, j] > a_ori and gradient[i, j] > b_ori:
                    flag = True

            elif angle[i, j] > 0:
                num_1 = (inter_win[0, 2] - inter_win[1, 2]) * angle[i, j] + inter_win[1, 2]
                num_2 = (inter_win[2, 0] - inter_win[1, 0]) * angle[i, j] + inter_win[1, 0]
                if gradient[i, j] > num_1 and gradient[i, j] > num_2:
                    flag = True

            elif angle[i, j] < 0:
                num_1 = (inter_win[1, 0] - inter_win[0, 0]) * angle[i, j] + inter_win[1, 0]
                num_2 = (inter_win[1, 2] - inter_win[2, 2]) * angle[i, j] + inter_win[1, 2]
                if gradient[i, j] > num_1 and gradient[i, j] > num_2:
                    flag = True

            if flag:
                yizhi_img[i, j] = gradient[i, j]
    plt.figure(3)
    plt.imshow(yizhi_img.astype(np.uint8), cmap='gray')
    plt.axis('off')

    zhan = []
    max_threshold = np.average(yizhi_img) * 5
    min_threshold = max_threshold / 3
    for i in range(1, yizhi_img.shape[0]-1):
        for j in range(1, yizhi_img.shape[1]-1):
            if yizhi_img[i, j] >= max_threshold:
                yizhi_img[i, j] = 255
            elif yizhi_img[i, j] < min_threshold:
                yizhi_img[i, j] = 0
            else:
                zhan.append([i, j])

    while len(zhan) != 0:
        temp1, temp2 = zhan.pop()
        a = yizhi_img[temp1-1:temp1+2, temp2-1:temp2+2]
        if a[0, 0] == 255 or a[0, 1] == 255 or a[0, 2] == 255 or a[1, 0] == 255 or a[1, 2] == 255 or a[2, 0] == 255 or a[2, 1] == 255 or a[2, 2] == 255:
            yizhi_img[temp1, temp2] = 255

    for i in range(yizhi_img.shape[0]):
        for j in range(yizhi_img.shape[1]):
            if yizhi_img[i, j] != 0 and yizhi_img[i, j] != 255:
                yizhi_img[i, j] = 0

    plt.figure(4)
    plt.imshow(yizhi_img.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()








