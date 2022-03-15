# 目标：手写实现均值哈希与差值哈希算法

import cv2
import numpy as np


def ahash(img):
    """
    均值哈希
    :param img: 图像源
    :return: 哈希字符串
    """
    # interpolation - 插值方法。共有5种：
    # INTER_NEAREST - 最近邻插值法
    # INTER_LINEAR - 双线性插值法（默认）
    # INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）
    # INTER_CUBIC - 基于4x4像素邻域的3次插值法
    # INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    gray = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_CUBIC)  # 缩放为8*8
    avg = np.mean(gray)
    # print(avg)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dhash(img):
    """
    差值哈希
    :param img: 图像源
    :return: 哈希字符串
    """
    # interpolation - 插值方法。共有5种：
    # INTER_NEAREST - 最近邻插值法
    # INTER_LINEAR - 双线性插值法（默认）
    # INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）
    # INTER_CUBIC - 基于4x4像素邻域的3次插值法
    # INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    gray = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_CUBIC)  # 缩放为9*8，要做差值
    # print(gray.shape)  # 值为(8,9)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def hanmingDist(hash1, hash2):
    """
    :param hash1: 比较哈希值1
    :param hash2: 比较哈希值2
    :return: 差异值：-1为输入错误，其他值为相似度，越小越相似
    """
    cnt = 0
    # 参数检查，要求哈希串长度相同
    if len(hash1) != len(hash2):
        return -1
    # 遍历哈希串，如果hash相同位置的元素不一致,则cnt+1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            cnt += 1
    return cnt


if __name__ == '__main__':
    # 读入图
    img1 = cv2.imread("lenna.png")
    img2 = cv2.imread("lenna_noise.png")
    ahash1 = ahash(img1)
    ahash2 = ahash(img2)
    print("ahash1", ahash1)
    print("ahash2", ahash2)
    cnt = hanmingDist(ahash1, ahash2)
    print('均值哈希算法相似度(汉明距离)：', cnt)

    dhash1 = dhash(img1)
    dhash2 = dhash(img2)
    print("dhash1", dhash1)
    print("dhash2", dhash2)
    cnt = hanmingDist(dhash1, dhash2)
    print('差值哈希算法相似度(汉明距离)：', cnt)
