#!/usr/bin/env python 
# coding:utf-8
import cv2


# 均值哈希算法
def aHash(img_gray):
    # 将图像缩至到8*8
    img = cv2.resize(img_gray, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 遍历图像,求出其像素和
    pixel_sum = 0
    hash_str = ""
    for i in range(8):
        for j in range(8):
            pixel_sum = pixel_sum + img[i, j]
    # 求均值
    avg = pixel_sum / (8 * 8)
    # 遍历判断像素,如果像素值大于均值,hash_str累加1,否则累加0
    for i in range(8):
        for j in range(8):
            if img[i, j] > avg:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


# 差值哈希算法
def dHash(img_gray):
    # 将图像缩放至8*9
    img = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 遍历其像素,如果前一个像素大于后一个像素,则hash_str累加1,否则累加0
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if img[i, j] > img[i, j + 1]:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


# 比较两个哈希值相似度
def compare_hash_value(hash_value1, hash_value2):
    # 进行判断,如果传入的哈希值长度不同,则返回 - 1
    if len(hash_value1) != len(hash_value2):
        return -1
    cnt = 0
    # 如果hash相同位置的元素不一致,则cnt+1
    for i in range(len(hash_value1)):
        if hash_value1[i] != hash_value2[i]:
            cnt += 1
    return cnt


if __name__ == '__main__':
    # 读入灰度图
    img_gray = cv2.imread("lenna.png", 0)
    img_gray2 = cv2.imread("lenna_noise.png", 0)
    aHash_value = aHash(img_gray)
    aHash_value2 = aHash(img_gray2)
    # 两个哈希值的相似度
    cnt = compare_hash_value(aHash_value, aHash_value2)
    print("均值哈希相似度：\n", cnt)
    dHash_value = dHash(img_gray)
    dHash_value2 = dHash(img_gray2)
    cnt = compare_hash_value(dHash_value, dHash_value2)
    print("差值哈希相似度：\n", cnt)
