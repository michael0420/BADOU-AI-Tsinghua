import cv2
import numpy as np

def MeanHash(img):
    """
    均值hash算法
    """
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resize_img = cv2.resize(gray_img,(8,8))
    mean = np.mean(resize_img)
    resize_img[resize_img<=mean] = 0
    resize_img[resize_img>mean] = 1
    hash_map = []
    for i in resize_img:
        hash_map.extend(list(i))
        
    hash_map = [str(i) for i in hash_map]
    hash_str = ''.join(hash_map)
    return hash_str


def DifferHash(img):
    """
    均值hash算法
    """
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resize_img = cv2.resize(gray_img,(9,8))
    hash_map = np.zeros((8,8),np.int16)
    for i in range(8):
        hash_map[:,i] = resize_img[:,i]>resize_img[:,i+1]
    hash_list = []
    for i in hash_map:
        hash_list.extend(list(i))
    hash_str = [str(i) for i in hash_list]
    hash_str = ''.join(hash_str)

    return hash_str



if __name__=='__main__':
    
    img = cv2.imread('../lenna.png')
    hash_mean = MeanHash(img)
    print('MeanHash:',hash_mean)
    hash_differ = DifferHash(img)
    print('DifferHash',hash_differ)

    
