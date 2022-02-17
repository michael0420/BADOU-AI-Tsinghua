import cv2
import numpy as np


#均值哈希算法
#步骤
#1. 缩放：图片缩放为8*8，保留结构，除去细节。
#2. 灰度化：转换为灰度图。
#3. 求平均值：计算灰度图所有像素的平均值。
#4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
#5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
#6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似。

def average_Hash(img):
    # 1. 缩放：图片缩放为8*8，保留结构，除去细节。
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 2. 灰度化：转换为灰度图。
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. 求平均值：计算灰度图所有像素的平均值。
    # 遍历图像,求出其像素和
    pixel_sum = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            pixel_sum = pixel_sum + img_gray[i, j]
    # 求均值
    avg = pixel_sum / (8 * 8)
    # 4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#差值哈希算法

#步骤
#1. 缩放：图片缩放为8*9，保留结构，除去细节。
#2. 灰度化：转换为灰度图。
#3. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，八个差值，有8行，总共64位
#4. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
#5. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似。

def differential_Hash(img):
    # 1. 缩放：图片缩放为8*8，保留结构，除去细节。
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    print(img.shape) #第一个元素表示矩阵行数，第二个元组表示矩阵列数，第三个元素是3，表示像素值由光的三原色组成。
    # 2. 灰度化：转换为灰度图。
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，八个差值，有8行，总共64位
    hash_str=''
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1 = average_Hash(img1)
hash2 = average_Hash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = differential_Hash(img1)
hash2 = differential_Hash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)
