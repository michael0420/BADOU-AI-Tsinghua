#!/usr/bin/env python 
# coding:utf-8
import random

import cv2


def add_salt_and_pepper_noise(src, percentage):
    copy_img = src.copy()
    # 图像行、列
    height, width = copy_img.shape[0], copy_img.shape[1]
    # 需要加椒盐噪声的像素数
    noise_num = int(height * width * percentage)
    for i in range(noise_num):
        # 随机取像素位置
        rand_x = random.randint(0, height - 1)
        rand_y = random.randint(0, width - 1)
        # 获取随机数
        random_data = random.random()
        # 如果随机数小于0.5,将该像素值置为255,反之为0
        if random_data < 0.5:
            copy_img[rand_x, rand_y] = 255
        else:
            copy_img[rand_x, rand_y] = 0
    return copy_img


if __name__ == '__main__':
    # 百分比
    percentage = 0.3
    # 读入灰度图
    img = cv2.imread("lenna.png", 0)
    noise_img = add_salt_and_pepper_noise(img, percentage)
    cv2.imshow("salt_and_pepper_noise", noise_img)
    cv2.imshow("src", img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
