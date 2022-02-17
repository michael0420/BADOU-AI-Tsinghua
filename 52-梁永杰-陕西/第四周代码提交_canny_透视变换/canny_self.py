'''
@auther:Jelly
用于实现Canny边缘检测算法
sobel实现
函数接口：
    def Gauss_filter_self(img)
    函数用途：给灰度图像进行高斯滤波
    函数参数
        img：灰度图像

    def Sobel_self(img,dx=False,dy=False)
    函数用途：进行Sobel算子的运算
    函数参数：
        img:接受灰度图像数据
        dx:x方向上求导，求导阶数为一阶
        dy:y方向上求导，求导阶数为一阶

    def NMS_self(img,img_x,img_y)
    函数用途：非极大值抑制
    函数参数：
        img: 边缘检测后的图像
        img_x:x方向上边缘检测后的图像
        img_y:x方向上边缘检测后的图像

    def Double_threshold_self(img,thresholdmin,thresholdmax)
    函数用途：双阈值检测，连接边缘（二值化，非极大值边缘的点灰度值为0）
    函数参数：
        img:灰度图
        thresholdmin:小阈值
        thresholdmax:大阈值

    def canny_self(img,thresholdmin,thresholdmax)
    函数用途：实现Canny图像边缘检测
    函数参数：
        img：带处理的灰度图像
        thresholdmin：双阈值限制的低阈值
        thresholdmax：双阈值限制的高阈值


'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def Sobel_self(img,dx=False,dy=False):
    '''
    函数用途：进行Sobel算子的运算
    函数参数：
        img:接受灰度图像数据
        dx:x方向上求导，求导阶数为一阶
        dy:y方向上求导，求导阶数为一阶
    '''
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    h, w = img.shape
    img_sobel = np.zeros((h,w))

    if dx==True:
        sobel_kernel = [[-1,0,1],[-2,0,2],[-1,0,1]]
    if dy==True:
        sobel_kernel = [[1,2,1],[0,0,0],[-1,-2,-1]]
    for i in range(h):
        for j in range(w):
            img_sobel[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel)
    return img_sobel

def Gaussian_filter_self(img):
    '''
    函数用途：给灰度图像进行高斯滤波
    函数参数
        img：灰度图像
    '''
    sigma =0.5
    dim = int(np.round(6 * sigma + 1)) #根据标准差，求高斯核的维度是几乘几的
    if dim % 2 == 0: #保证维度是奇数
        dim += 1
    Gaussian_filter = np.zeros([dim,dim])
    tmp = [i-dim//2 for i in range(dim)] # 生成一个序列
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):           #利用高斯公式构建高斯滤波矩阵
        for j in range(dim):
            Gaussian_filter[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter /Gaussian_filter.sum() #将滤波核归一
    dx,dy = img.shape
    img_new = np.zeros(img.shape)
    tmp = dim//2
    img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter)
    #return img_new.astype(np.int16)
    return img_new

def NMS_self(img,img_x,img_y):
    '''
    函数用途：非极大值抑制
    函数参数：
        img: 边缘检测后的图像
        img_x:x方向上边缘检测后的图像
        img_y:x方向上边缘检测后的图像
    '''
    angle = img_y/img_x
    dx,dy = img.shape
    img_NMS = np.zeros((dx,dy))
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            flag = True                 #在8个邻域内，是否要将该点的像素抹去
            temp = img[i-1:i+2,j-1:j+2] #梯度幅值的8邻域矩阵
            # 角度线落在x轴方向上
            if angle[i,j] <= -1  :      #使用现行插值法判断是否进行抑制
                num_1 = (temp[0,1] - temp[0,0]) / (angle[i,j] + temp[0,1])   # 保证求出的斜率与angle的符号方向相同
                num_2 = (temp[2,1] - temp[2,2]) / (angle[i,j] + temp[2,1])
                if not (img[i,j]>num_1 and img[i,j]>num_2):
                    flag = False
            elif angle[i,j] >= 1:
                num_1 = (temp[0,2] - temp[0,1]) / (angle[i,j] + temp[0,1])  # 保证求出的斜率与angle的符号方向相同
                num_2 = (temp[2,1] - temp[2,0]) / (angle[i,j] + temp[2,1])
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            # 角度线落在y轴方向上
            elif angle[i,j] < 0:
                num_1 = (temp[1,0] - temp[0,0]) / (angle[i,j] + temp[1,0])  # 保证求出的斜率与angle的符号方向相同
                num_2 = (temp[1,2] - temp[2,2]) / (angle[i,j] + temp[1,2])
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif angle[i,j] > 0:
                num_1 = (temp[2,0] - temp[1,0]) / (angle[i,j] + temp[1,0])  # 保证求出的斜率与angle的符号方向相同
                num_2 = (temp[0,2] - temp[1,2]) / (angle[i,j] + temp[1,2])
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            if flag:
                img_NMS[i,j] = img[i,j]
    return img_NMS

def Double_threshold_self(img,thresholdmin,thresholdmax):
    '''
    函数用途：双阈值检测，连接边缘（二值化，非极大值边缘的点灰度值为0）
    函数参数：
        img:灰度图
        thresholdmin:小阈值
        thresholdmax:大阈值
    '''
    h,w = img.shape
    zhan = []
    for i in range(1,h-1):             #不考虑外圈的像素,这样才能将周围的像素点进行判断是否为边缘
        for j in range(1,w-1):
            if img[i,j] >= thresholdmax:
                img[i,j] = 255
                # 记录所有已确认边缘的坐标
                # 为判断其他弱边缘的像素提供坐标依据
                zhan.append([i,j])
            elif img[i,j] <= thresholdmin:
                img[i,j] = 0

    while not len(zhan) == 0:
        #通过强边缘进一步将其他若边缘进行区分
        #   若在一个已确认为边缘处的像素点周围的像素，有在双阈值范围的话，也被列为边缘，
        #   在进一步判断其周围的点是否有能归为边缘的像素点
        temp_1,temp_2 = zhan.pop()  #出栈
        a = img[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
        # 若其中周围8个点有一个点在阈值内则为边缘点，否则不是边缘点
        if (a[0,0]<thresholdmax) and (a[0,0]>thresholdmin):
            img[temp_1-1,temp_2-1] = 255 # 标记为边缘
            zhan.append([temp_1-1,temp_2-1]) #进栈

        if (a[0,1]<thresholdmax) and (a[0,1]>thresholdmin):
            img[temp_1-1,temp_2] = 255 # 标记为边缘
            zhan.append([temp_1-1,temp_2]) #进栈

        if (a[0,2]<thresholdmax) and (a[0,2]>thresholdmin):
            img[temp_1-1,temp_2+1] = 255 # 标记为边缘
            zhan.append([temp_1-1,temp_2+1]) #进栈

        if (a[1,0]<thresholdmax) and (a[1,0]>thresholdmin):
            img[temp_1,temp_2-1] = 255 # 标记为边缘
            zhan.append([temp_1,temp_2-1]) #进栈

        if (a[1,2]<thresholdmax) and (a[1,2]>thresholdmin):
            img[temp_1,temp_2+1] = 255 # 标记为边缘
            zhan.append([temp_1,temp_2+1]) #进栈

        if (a[2,0]<thresholdmax) and (a[2,0]>thresholdmin):
            img[temp_1+1,temp_2-1] = 255 # 标记为边缘
            zhan.append([temp_1+1,temp_2-1]) #进栈

        if (a[2,1]<thresholdmax) and (a[2,1]>thresholdmin):
            img[temp_1+1,temp_2] = 255 # 标记为边缘
            zhan.append([temp_1+1,temp_2]) #进栈

        if (a[2,2]<thresholdmax) and (a[2,2]>thresholdmin):
            img[temp_1+1,temp_2+1] = 255 # 标记为边缘
            zhan.append([temp_1+1,temp_2+1]) #进栈

    for i in range(h):
        for j in range(w):
            if img[i,j] != 0 and img[i,j] != 255:
                img[i,j] = 0

    return img


def canny_self(img,thresholdmin,thresholdmax):
    '''
    函数用途：实现Canny图像边缘检测
    函数参数：
        img：带处理的灰度图像
        thresholdmin：双阈值限制的低阈值
        thresholdmax：双阈值限制的高阈值
    '''
    dx,dy = img.shape
    #图像进行高斯滤波
    img_Gaussian = Gaussian_filter_self(img)
    #检测图像中的水平、垂直和对角边缘
    img_sobel_X = Sobel_self(img_Gaussian,dx=True,dy=False) # X方向上进行边缘检测
    img_sobel_Y = Sobel_self(img_Gaussian,dx=False,dy=True) # 向上进行边缘检测
    img_sobel = np.zeros((dx,dy))
    for i in range(dx):
        for j in range(dy):
            img_sobel[i,j] = np.sqrt(img_sobel_X[i,j]**2+img_sobel_Y[i,j]**2)
    img_NMS = NMS_self(img_sobel,img_sobel_X,img_sobel_Y)
    img_DT = Double_threshold_self(img_NMS,thresholdmin,thresholdmax)
    return img_DT


if __name__ == '__main__':

    img = cv2.imread('lenna.png')
    img_gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    img_canny = canny_self(img_gary,50,255)

    #plt.imshow(img_canny.astype(np.uint8),cmap='gray')
    plt.imshow(img_canny,cmap='gray')
    plt.show()