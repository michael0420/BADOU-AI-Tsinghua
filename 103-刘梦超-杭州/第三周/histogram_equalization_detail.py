#!/usr/bin/env python 
# coding:utf-8

import cv2


# 遍历通道,统计每个灰度值出现的次数
def get_pix_cnt_dict(item_channel_pixel):
    pix_cnt_dict = dict()
    for item in item_channel_pixel:
        if item in pix_cnt_dict:
            pix_cnt_dict[item] += 1
        else:
            pix_cnt_dict[item] = 0
    return pix_cnt_dict


# 将原像素值映射为新的值,保存在字典中
def get_new_pixel_dict(pix_cnt_dict, height, width):
    sum = 0.0
    new_pixel_dict = dict()
    # 遍历灰度值,进行映射
    for pixel in range(256):
        # 当前灰度值出现的次数
        cnt = pix_cnt_dict.get(pixel)
        if cnt is None:
            cnt = 0
        # 当前像素个数除以总像素值
        pi = cnt / (height * width)
        # 进行累加
        sum += pi
        # 映射出新的像素
        mapper_pixel = sum * 256 - 1
        # 如果像素为负值,取0
        if mapper_pixel < 0:
            mapper_pixel = 0
        else:
            # 四舍五入
            mapper_pixel = int(mapper_pixel + 0.5)
        # 映射后的像素值,保存在字典中
        new_pixel_dict[pixel] = mapper_pixel
    return new_pixel_dict


# 将映射得到的新元素替换到矩阵中
def mapper_array_pixel(new_pixel_dict, item_channel_pixel, height, width):
    for i in range(len(item_channel_pixel)):
        if item_channel_pixel[i] in new_pixel_dict:
            item_channel_pixel[i] = new_pixel_dict.get(item_channel_pixel[i])
    # 将数据转为n*m矩阵
    transform_channel_pixel = item_channel_pixel.reshape(height, width)
    return transform_channel_pixel


# 计算直方图
def calc_hist(channel_img):
    # 获取行，列
    height, width = channel_img.shape
    # 将数据打平
    item_channel_pixel = channel_img.ravel()
    # 通道数据字典
    pix_cnt_dict = get_pix_cnt_dict(item_channel_pixel)
    # 映射后的新元素保存在字典中
    new_pixel_dict = get_new_pixel_dict(pix_cnt_dict, height, width)
    # 替换原矩阵中的元素
    item_channel_pixel = mapper_array_pixel(new_pixel_dict, item_channel_pixel, height, width)
    return item_channel_pixel


if __name__ == '__main__':
    # 读入原图
    img = cv2.imread("lenna.png")
    # 读入的是彩色图像,需要对该图像进行通道分割
    (b_img, g_img, r_img) = cv2.split(img)
    # 对每个通道进行均衡化处理
    b_hist = calc_hist(b_img)
    g_hist = calc_hist(g_img)
    r_hist = calc_hist(r_img)
    # 合并为一个通道
    result = cv2.merge((b_hist, g_hist, r_hist))
    # 彩色图均值化后的结果
    cv2.imshow("hist_equalization_detail_img", result)
    # 原图
    cv2.imshow("src_img", img)
    cv2.waitKey(0)
