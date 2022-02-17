import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def normal_picture(pic_adr):

    dimension = input("你想要处理彩图还是灰度图，彩图输入3，灰图输入2：\n")
    if dimension == '3':
        img = cv.imread(pic_adr)
    elif dimension == '2':
        img = cv.imread(pic_adr, 0)
    else:
        print("拜拜！\n")
        sys.exit()

#####################################
    if len(img.shape) == 3:#彩色图
        h,w,d = img.shape
        oneGray_ave = h * w / 256
        (b, g, r) = cv.split(img)

        flag = 0
        bb = np.zeros_like(b)
        gg = np.zeros_like(b)
        rr = np.zeros_like(b)

        for dim in (b,g,r):
            num = dim.flatten().tolist()
            count = [0 for i in range(0, 256)]
            for i in range(0, 256):
                count[i] += num.count(i)
            equa = [0 for i in range(0, 256)]
            equa[0] = (count[0] / oneGray_ave)
            for i in range(1, 256):
                equa[i] = equa[i - 1] + (count[i] / oneGray_ave)
            equa = np.trunc(equa)
            equa = (equa - 1).astype(int)
            equa = abs(equa)
            img2 = np.zeros_like(dim)
            for i in range(0, h):
                for j in range(0, w):
                    img2[i][j] = equa[round(dim[i][j])]

            if flag == 0:
                bb = img2
            elif flag == 1:
                gg = img2
            else:
                rr = img2
            flag += 1
        dd = cv.merge((bb,gg,rr))
        cv.imshow("3dimension equalize",dd)
        cv.waitKey(0)

#################################################
    elif len(img.shape) == 2:
        plt.subplot(1,2,1)
        imgg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(imgg)

        # hist = cv.calcHist([img.ravel()],[0],None,[256],[0,256])
        # plt.subplot(2, 2, 2)
        # plt.imshow(hist)

        #print(hist)
        h, w = img.shape
        #plt.hist(hist.ravel(),256,(0,256))#plt.plot(hist)#这玩意画出来是折线
        #plt.show()
        oneGray_ave = h*w/256
        num = img.flatten().tolist()
        #print(type(num))
        #逐像素便利
        count = [0 for i in range(0,256)]
        for i in range(0,256):
            count[i] += num.count(i)
        #print(count)#能行
        #求均衡化后的统计列表
        equa = [0 for i in range(0, 256)]

        equa[0] = (count[0]/oneGray_ave)
        for i in range(1, 256):
            equa[i] = equa[i-1]+(count[i]/oneGray_ave)


        equa = np.trunc(equa)
        equa = (equa-1).astype(int)
        equa = abs(equa)
        #print(equa)
        img2 = np.zeros_like(img)
        for i in range(0,h):
            for j in range(0,w):
                img2[i][j] = equa[round(img[i][j])]
        #print(img2)
        img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.show()

class PCA_detail():
    def __init__(self,Array,K,Num):
        self.K = K
        self.a_initial = Array
        self.N = Num
        #这里调用函数要写self
        self.a_centralize = self.centralize()
        self.a_cov = self.cov()
        self.a_change = self.change()
        self.a_end = self.dot()

    def centralize(self):
        arr = self.a_initial
        mean = [np.mean(aa) for aa in np.transpose(arr)]
        #print(mean)
        cen = arr - mean
        #print("centralize:",cen)
        return cen

    def cov(self):
        arr = self.a_centralize
        #print("centralize:\n",arr)
        ccov = np.dot(np.transpose(arr),arr)/(self.N-1)#输出3x3的矩阵
        #print("cov:",ccov)
        return ccov

    def change(self):#三维变二维矩阵的算子？
        arr = self.a_cov
        fea,fea_vec = np.linalg.eig(arr)#直接硬套库函数
        #print('特征值',fea)
        #print('特征向量:\n',fea_vec)
        #返回值按大小顺序排后，的位置下标，下标存在新矩阵中
        index_H2L = (-1*fea).argsort()#不加-1就是从小到大[1 2 0]
        #print(index_H2L)
        #fea_vec_T = fea_vec.T
        fea_vec_lower = [fea_vec[:,i] for i in range(self.K-1)]
        #print(np.transpose(fea_vec_lower))
        return np.transpose(fea_vec_lower)

    def dot(self):
        arr_change = self.a_change#3x2
        arr_initial = self.a_initial#5x3

        arr_end = np.dot(arr_initial,arr_change)
        print('原矩阵:\n',self.a_initial)
        print('降维完成：\n',arr_end)

if __name__ == '__main__':
    #pic_adr = input("输入图片的名称（本文件夹存在）")
    pic_adr = 'lenna.png'
    print("请输入数字进行操作\n1：直方图均衡化\n2：PCA降维操作")
    op = input("操作是：")

    if op == '1':
        normal_picture(pic_adr)
    elif op == '2':
        Array = np.array([
            [2.443,5.665,8.587],
            [1.576,6.466,8.356],
            [6.754,6.357,1.345],
            [5.674,6.257,6.256],
            [8.356,9.345,2.456]
        ])
        Num = np.shape(Array)[0]
        K = np.shape(Array)[1]
        PCA_detail(Array,K,Num)