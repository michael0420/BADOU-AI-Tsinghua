import cv2
import numpy as np

#定义感知哈希
def phash(img):
    #step1：调整大小
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img.astype(np.float32)

    #step2:离散余弦变换
    img=cv2.dct(img)
    img=img[0:8,0:8]
    sum=0.
    hash_str=''

    #step3:计算均值
    # avg = np.sum(img) / 64.0
    for i in range(8):
        for j in range(8):
            sum+=img[i,j]
    avg=sum/64

    #step4:获得哈希
    for i in range(8):
        for j in range(8):
            if img[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#计算汉明距离
def hmdistance(hash1,hash2):
    num=0
    assert len(hash1)==len(hash2)
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            num+=1
    return num


img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna_noise.png')

hash1=phash(img1)
hash2=phash(img2)

print(hash1)
print(hash2)

dist=hmdistance(hash1,hash2)
print('均值哈希算法相似度：',dist)


