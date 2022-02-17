import cv2
import numpy


'''
哈希算法
'''

#1均值哈希算法
'''
1. 缩放：图片缩放为8*8，保留结构，除去细节。
2. 灰度化：转换为灰度图。
3. 求平均值：计算灰度图所有像素的平均值。
4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不
相同位数越少，图片越相似。
'''
def average_hash(src_img):
    #1.1.缩放图片为8*8
    dst_img = cv2.resize(src_img,(8,8),interpolation=cv2.INTER_CUBIC)
    #cv2.imshow(dst_img)
    #1.2.转灰度
    gray_img = cv2.cvtColor(dst_img,cv2.COLOR_BGR2GRAY)
    #1.3.求平均值
    sum = 0
    hash_init = ''
    for i in range(8):
        for j in range(8):
            sum += gray_img[i,j]
    aver_gray = sum / (8*8)
    #1.4.比较
    #1.5.生成哈希
    #大于平均置一，否则置零
    for i in range(8):
        for j in range(8):
            if gray_img[i,j]>aver_gray:
                hash_init += '1'
            else:
                hash_init += '0'
    return hash_init

#差值哈希算法
'''
1. 缩放：图片缩放为8*9，保留结构，除去细节。
2. 灰度化：转换为灰度图。
3. 求平均值：计算灰度图所有像素的平均值。 ---这步没有，只是为了与均值哈希做对比
4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
八个差值，有8行，总共64位
5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样
的，不相同位数越少，图片越相似。
'''
def divide_hash(src_img):
    # 2.1.缩放图片为8*8
    dst_img = cv2.resize(src_img, (8, 9), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow(dst_img)
    # 2.2.转灰度
    gray_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    #2.4.比较
    #2.5.生成哈希
    #前一个像素灰度值大于后一个灰度值置一，否则置零
    hash_init = ''
    for i in range(8):
        for j in range(8):
            if gray_img[i,j]>gray_img[i+1,j]:
                hash_init+='1'
            else:
                hash_init += '0'

    return hash_init

#3. 对比hash值
def compare_hash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

img1 = cv2.imread('F:/cycle_gril/lenna.png')
img2 = cv2.imread('F:/cycle_gril/lenna_noise.png')
value_hash_ave1 = average_hash(img1)
value_hash_ave2 = average_hash(img2)
print(value_hash_ave1)
print(value_hash_ave2)

print('均值哈希算法相似度',compare_hash(value_hash_ave1,value_hash_ave2))


value_hash_ave1 = divide_hash(img1)
value_hash_ave2 = divide_hash(img2)
print(value_hash_ave1)
print(value_hash_ave2)

print('差值哈希算法相似度',compare_hash(value_hash_ave1,value_hash_ave2))


cv2.waitKey(0)
