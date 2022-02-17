import cv2
import numpy as np

# 均值哈希
def aHash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)  # 缩放为8*8
    '''
    cv2.resize(InputArray,Size, interpolation)
    InputArray:输入图片
    Size：大小
    interpolation:   插值方式
          INTER_NEAREST 最近邻插值
          INTER_LINEAR  双线性插值（默认设置）
          INTER_AREA  使用像素区域关系进行重采样。
          INTER_CUBIC  4x4像素邻域的双三次插值
          INTER_LANCZOS48x8像素邻域的Lanczos插值
    '''  # cv2.resize 描述
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 转化为灰度图
    s=0
    hash_str=''
    for i in range(8):
        for j in range(8):
            s+=gray[i,j]
    avg=s/64
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

def dHash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)  # 缩放为8*9
    # 此处 (9,8) 为9列8行
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(gray)
    hash_str=''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

def cmpHash(hash1,hash2):
    n=0
    if len(hash1)!=len(hash2):  # 长度不同传参出错
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n+=1
    return n

def test():
    img1=cv2.imread('lenna.png')
    img2=cv2.imread('lenna_noise.png')
    hash1=aHash(img1)
    hash2=aHash(img2)
    print('',hash1,'\n',hash2)
    n=cmpHash(hash1,hash2)
    print('均值哈希算法相似度:',n)

    hash1 = dHash(img1)
    hash2 = dHash(img2)
    print('',hash1, '\n',hash2)
    n = cmpHash(hash1,hash2)
    print('差值哈希算法相似度:', n)

test()