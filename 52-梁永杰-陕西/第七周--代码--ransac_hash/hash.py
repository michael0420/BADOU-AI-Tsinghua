import numpy as np
import cv2

def average_hash(img):
    '''
    函数用途：用于实现均值哈希算法
    函数参数：输入图像
    返回值：哈希值
    '''
    #将图片缩放为8*8
    img2 = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)  # 三次样条插值
    #会度化
    img_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #比较
    hash_str = ''  # 建立哈希质问数据
    sum = np.sum(img_gray) # 像素和
    avg = sum/64
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str   # 得到 64位hash值

def diff_hash(img):
    '''
    函数用途：实现差值哈希算法
    函数参数：输入图像
    返回值：哈希值

    '''
    # 图片缩放 (要实现在行进行差值，及行大小要为9)
    img1 = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC) # 三次样条插值
    # 图片灰度化
    img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 在每一行处前一项与后一项进行差值判断，大于0为1 ，小于0为0
    for i in range(8):
        for j in range(8):
            if img_gray[i,j] > img_gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str





def hash_cmp(hash_str1,hash_str2):
    '''
    函数用途：实现哈希值比较
    函数参数：任意两个哈希值
    返回值：两个哈希值的汉明距离
    '''
    n = 0  # 初始化汉明长距离
    # 哈希值长度不统一，及不是代表同一类数据，无可比性
    if len(hash_str1) != len(hash_str2):
        return -1
    else:
        for i in range(len(hash_str2)):
            if hash_str1[i] != hash_str2[i]:
                n = n + 1
        return n




img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hash1= average_hash(img1)
hash2= average_hash(img2)
cmp_n1 = hash_cmp(hash1,hash2)
print('均值哈希算法')
print(hash1,hash2)
print('差值哈希算法汉明距离：',cmp_n1,'\n')


hash1 = diff_hash(img1)
hash2 = diff_hash(img2)
cmp_n2 = hash_cmp(hash1,hash2)
print('差值哈希算法')
print(hash1,hash2)
print('差值哈希算法汉明距离：',cmp_n2,'\n')