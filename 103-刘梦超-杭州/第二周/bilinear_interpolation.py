#!/usr/bin/env python 
# coding:utf-8
import cv2
import numpy as np


# 双线性插值
def bilinear_interpolation_function(img, dst_size):
    # 获取原图像的行,列,通道
    height, width, channels = img.shape
    # 获取目标图像的行,列
    dst_h, dst_w = dst_size[1], dst_size[0]
    # 校验,如果原图和目标大小一样,直接返回原图浅拷贝对象
    if height == dst_h and width == dst_w:
        return img.copy()
    # 设置目标图像的规格
    dst_img = np.zeros((dst_h, dst_w, 3), img.dtype)
    # 原图像和目标图像的比例关系
    ratio_x, ratio_y = width / dst_w, height / dst_h
    # 每个通道都需要遍历赋值
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 为了减少误差,需要对两个图像做中心对称
                src_x = (dst_x + 0.5) * ratio_x - 0.5
                src_y = (dst_y + 0.5) * ratio_y - 0.5
                # 取四个点的坐标,用做插值计算
                src_x0 = int(src_x)
                src_y0 = int(src_y)
                # 防止边界溢出,需要取min
                src_x1 = min(src_x0 + 1, width - 1)
                src_y1 = min(src_y0 + 1, height - 1)
                # 套入公式进行计算
                # x方向做插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                # y方向做插值
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # 整合
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    # 读取原图
    img = cv2.imread("lenna.png")
    # 设置目标图片的大小
    dst_size = (700, 700)
    # 处理目标图片
    dst_img = bilinear_interpolation_function(img, dst_size)
    # 显示最终结果
    cv2.imshow("dst_img", dst_img)
    cv2.waitKey(0)
