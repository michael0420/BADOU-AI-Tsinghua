#coding = utf-8
import cv2
import numpy as np

def meanHash(img):
    imgr=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(imgr,cv2.COLOR_BGR2GRAY)
    mean_value=0
    for i in range(8):
        for j in range(8):
            mean_value+=gray[i,j]
            
    mean_value/=64
    hash_value=''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>mean_value :
                hash_value=hash_value+'1'
            else:
                hash_value=hash_value+'0'
                
    return hash_value
    
def diffHash(img):
    imgr=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(imgr,cv2.COLOR_BGR2GRAY)
    hash_value=''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_value=hash_value+'1'
            else:
                hash_value=hash_value+'0'
    return hash_value
        

def cmpHash(hash1,hash2):
    if len(hash1)!=len(hash2):
        return -1
    n=0
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n += 1
    return n       

if __name__=='__main__':
    img1=cv2.imread('lenna.png')
    img2=cv2.imread('lenna_noise.png')
    hash1= meanHash(img1)
    hash2= meanHash(img2)
    print(hash1)
    print(hash2)
    n=cmpHash(hash1,hash2)
    print('均值哈希算法相似度：',n)
     
    hash3= diffHash(img1)
    hash4= diffHash(img2)
    print(hash3)
    print(hash4)
    n=cmpHash(hash3,hash4)
    print('差值哈希算法相似度：',n)
