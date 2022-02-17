import cv2
import numpy as np


#均值哈希算法
def aHash(img):

    #缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)

    #转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #求均值 和 hash_str
    avg = 0
    hash_str = ""
    for i in range(8):
        for j in range(8):
            avg += gray[i,j]

    #平均值
    avg = avg / 64

    #灰度值大于均值为1 小于均值为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

#插值算法
def dHash(img):

    #缩放 成8 * 9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)

    #灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #hash_str 每行前一个像素大于后一个像素为1， 反之为0
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


#hash 对比
def cmpHash(hash1, hash2):
    n = 0
    #hash长度不同则返回-1，代表传参出错
    if len(hash1) != len(hash2):
        return -1
    #便利判断
    for i in range(len(hash1)):
        #不想等则n 计数+1, n最终为相似度
        if hash1[i] != hash2[i]:
            n += 1
    return n

img1 = cv2.imread('../../../../../BaiduNetdiskDownload/lenna.png')
img2 = cv2.imread('../../../../../BaiduNetdiskDownload/lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)

n = cmpHash(hash1, hash2)
print("均值算法相似度：", n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('插值算法：', n)

