#pca详细实现
#矩阵中心化----求矩阵协方差----求协方差矩阵的特征值和特征向量--求K阶降维转置矩阵--Z = XU 求降维矩阵


import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris

import numpy as np

class OURPCA(object):

    def __init__(self, X, K):
        self.X = X   #样本矩阵X
        self.K = K   #样本矩阵降维K阶    2
        self.centreX = []   #中心化矩阵
        self.C = []  #协方差矩阵C
        self.U = []  #降维转置矩阵U
        self.Z = []  #降维矩阵Z

        self.centreX = self.Centre()
        self.C = self.XFC()
        self.U = self.JWZZ()
        self.Z = self._Z()

    def Centre(self):    #中心化
        centreX = []
        #先求均值
        mean = np.array([np.mean(a)   for a in self.X.T])   #for a in self.X.T这个式子将矩阵进行了转置并求均值
        print("均值为\n", mean)
        centreX = self.X -mean  #中心化
        print("中心化矩阵为\n", centreX)
        return centreX

    def XFC(self):     #求协方差矩阵
        #行为样例，列为特征维度
        #求出样例个数
        ns = np.shape(self.centreX)[0]  #10
        C = np.dot(self.centreX.T , self.centreX) / (ns - 1) #协方差矩阵
        return C

    def JWZZ(self):  #先求协方差矩阵特征值和特征向量，再求降维转置矩阵
        #特征值和特征向量
        x , y = np.linalg.eig(self.C)   #特征值给x，特征向量给y
        ind = np.argsort(-1 * x)  #argsort函数返回的是数组值从小到大的索引值
        UT = [y[:, ind[i]]   for i in range(self.K)] #求出降维矩阵的转置矩阵
        U = np.transpose(UT)   #再次转置
        return U

    def _Z(self):   #Z = XU 求降维矩阵
        Z = np.dot(self.X, self.U)    #求得降维矩阵
        print("%d 阶降维矩阵Z为\n" % self.K, Z )
        return Z






if __name__ == '__main' :
    '10样本3特征的样本集, 行为样例，列为特征维度'
    print(1)
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    print(2)
    K = np.shape(X)[1] - 1
    print(3)
    pca = OURPCA(X, K)
    #print(pca.centreX)








'''

#直接引用pca算法对莺尾花数据进行降维处理


data_x, data_y = load_iris(return_X_y = True)   #下载莺尾花四维数据x以及标签y
data_pca = dp.PCA(n_components=2)   #加载pca算法，确定留下特征数为2，默认为1
pca_x = data_pca.fit_transform(data_x)    #对思维数据x进行主成分分析降维,由原先的四维降成二维
 #fit_transform对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值等等，然后对该数据进行转换tranform，从而实现数据的标准化、归一化等等
red_x, red_y = [], []
green_x, green_y = [], []
blue_x, blue_y = [], []

for i in range(len(pca_x)):    #pca_x里面包含着两列n行的数据
    if data_y[i] == 0 :
        red_x.append(pca_x[i][0])
        red_y.append(pca_x[i][1])
        print("red_x",red_x)
        print("red_y",red_y)
    elif data_y[i] == 1 :
        green_x.append(pca_x[i][0])
        green_y.append(pca_x[i][1])
    else:
        blue_x.append(pca_x[i][0])
        blue_y.append(pca_x[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(green_x,green_y, c='g', marker='.')
plt.scatter(blue_x, blue_y, c='b', marker='h')
plt.show()
'''

