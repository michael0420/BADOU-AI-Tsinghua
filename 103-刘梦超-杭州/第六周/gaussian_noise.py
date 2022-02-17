#!/usr/bin/env python 
# coding:utf-8
import random

import cv2


# 添加高斯噪声
def add_gaussian_noise(src, mean, sigma, percentage):
    new_img = src.copy()
    height, width = new_img.shape[0], new_img.shape[1]
    # 总像素数
    total_pixel = height * width
    # 需要加高斯噪声的像素数
    noise_num = int(total_pixel * percentage)
    for i in range(noise_num):
        # 随机取像素位置
        rand_x = random.randint(0, height - 1)
        rand_y = random.randint(0, width - 1)
        # 输出像素=输入像素+高斯随机数
        pixel_out = new_img[rand_x, rand_y] + random.gauss(mean, sigma)
        # 加上随机数后,新的像素值可能不在[0,255],需要进行判断
        if pixel_out > 255:
            pixel_out = 255
        if pixel_out < 0:
            pixel_out = 0
        new_img[rand_x, rand_y] = pixel_out
    return new_img


if __name__ == '__main__':
    # 均值
    mean = 2
    # 标准差
    sigma = 4
    # 百分比
    percentage = 0.6
    # 读入灰度图
    img = cv2.imread("lenna.png", 0)
    gaussian_noise_img = add_gaussian_noise(img, mean, sigma, percentage)
    cv2.imshow("gaussian_noise_img", gaussian_noise_img)
    cv2.imshow("src", img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
