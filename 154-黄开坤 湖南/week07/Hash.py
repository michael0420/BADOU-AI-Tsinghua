#coding:utf-8

import cv2 as cv


#均值哈希算法实现
def avgHash(img):
    #缩放成8x8
    img = cv.resize(img, (8,8), interpolation=cv.INTER_CUBIC)
    # print(img, img.shape)
    #转换成灰度图
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #给定初始时用于存放，hash_str,sum
    sum = 0
    hash_str = ''
    #像数值求和
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sum = sum + img_gray[i, j]
    #求均值
    avg = sum/(img.shape[0] * img.shape[1])
    #生成0/1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#插值哈希算法
def difHash(img):
    #
    img = cv.resize(img, (9, 8), cv.INTER_CUBIC)    #(宽x高x通道)
    #
    # print(img.shape)    #(高x宽x通道)(8x9x3)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(img_gray.shape)   #(8x9)(高x宽)
    hash_str = ''
    #
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]-1):
            if img_gray[i, j] > img_gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#哈希值对比
def compareHash(Hash1, Hash2):
    n = 0  # 统计相同位次数
    #判断哈希值的长度是否相等
    if len(Hash1) != len(Hash2):
        return -1
    #遍历哈希值
    for i in range(len(Hash1)):
        if Hash1[i] != Hash2[i]:
            n = n+1
    return n

img1 = cv.imread('lenna.png')
img2 = cv.imread('lenna_noise.png')
hash1 = avgHash(img1)
hash2 = avgHash(img2)
print(hash1,'\n',hash2)
compare = compareHash(hash1, hash2)
print('均值哈希算法相似度：', compare)

hash1 = difHash(img1)
hash2 = difHash(img2)
print(hash1,'\n',hash2)
compare = compareHash(hash1, hash2)
print('差值哈希算法相似度：', compare)
