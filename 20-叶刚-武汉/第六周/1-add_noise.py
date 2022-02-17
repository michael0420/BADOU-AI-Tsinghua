"""
@GiffordY
实现高斯噪声、椒盐噪声
"""

import random
import cv2
import copy


def add_gaussian_noise(src_img, mean=0, sigma=0.1, percentage=0.5):
    """
    给图像添加高斯噪声
    src_img：输入图像
    mean：高斯分布的均值参数，默认值0
    sigma：高斯分布的标准差参数，默认值0.1
    percentage：float型，高斯噪声占所有像素的比例，默认值0.5
    """
    # 给图像加噪声时，为了不改变原来的图像，需要使用深拷贝
    # OpenCV C++提供两种深拷贝方法：
    # 1、使用 dst_img = src_img.clone()
    # 2、使用 src_img.copyTo(dst_img)
    # 也可以使用copy库提供的deepcopy()方法
    dst_img = copy.deepcopy(src_img)
    noise_num = int(percentage * src_img.shape[0] * src_img.shape[1])
    for i in range(noise_num):
        # random.randint(a, b)函数生成[a, b]之间的随机整数
        row = random.randint(0, src_img.shape[0] - 1)
        col = random.randint(0, src_img.shape[1] - 1)
        # 在原像素值上添加高斯随机数
        dst_img[row, col] = src_img[row, col] + random.gauss(mean, sigma)
        # 对超出[0, 255]的像素进行处理
        if dst_img[row, col] < 0:
            dst_img[row, col] = 0
        elif dst_img[row, col] > 255:
            dst_img[row, col] = 255
    return dst_img


def add_salt_and_pepper_noise(src_img, percentage=0.01, salt_vs_pepper=0.5):
    """
    给图像添加椒盐噪声
    src_img：输入图像
    percentage：float型，椒盐噪声占所有像素的比例，默认值0.01
    salt_vs_pepper：float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
    """
    dst_img = copy.deepcopy(src_img)
    noise_num = int(percentage * src_img.shape[0] * src_img.shape[1])
    for i in range(noise_num):
        row = random.randint(0, src_img.shape[0] - 1)
        col = random.randint(0, src_img.shape[1] - 1)
        # random.random生成[0, 1)范围的随机浮点数
        # 生成盐噪声的概率为 salt_vs_pepper
        if random.random() < salt_vs_pepper:
            dst_img[row, col] = 255
        else:
            dst_img[row, col] = 0
    return dst_img


if __name__ == '__main__':
    raw_img = cv2.imread('../00-data/images/lenna.png', 0)
    gauss_noise_img = add_gaussian_noise(raw_img, 0, 10, 0.6)
    salt_pepper_noise_img = add_salt_and_pepper_noise(raw_img, 0.01, 0.6)

    cv2.imshow('raw_img', raw_img)
    cv2.imshow('gauss_noise_img', gauss_noise_img)
    cv2.imshow('salt_pepper_noise_img', salt_pepper_noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

