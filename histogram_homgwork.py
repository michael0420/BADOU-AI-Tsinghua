#直方图均衡化

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("lenna.png",1)   #0指导入图像为灰色，1指导入图像为彩色
#print(img)
#cv2.imshow("img", img)
#cv2.waitKey()


#灰度图均衡化
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #将图像转为灰度图
#print(gray)
#img_data = gray.ravel()   #将数组扁平化
#plt.hist(x,bins.....)函数用于绘制直方图，x 为传入数据，bins为直方图的柱数，默认10,
#plt.hist(img_data, 256)
#plt.rcParams['font.sans-serif'] = ['SimHei']   #正常显示中文
#plt.xlabel("256 boxs", fontsize=14)
#plt.ylabel("number",fontsize=14)
#plt.title("灰度图直方图")
#plt.show()

#opencv中直方图的均衡化函数为cv2.equalizeHist()
#灰度图的均衡化
img_equal = cv2.equalizeHist(gray)
img_equal_data = img_equal.ravel()
plt.hist(img_equal_data, bins=256)    #生成直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel("256 boxs", fontsize=14)
plt.ylabel("number",fontsize=14)
plt.title("灰度图直方图")
image = np.hstack((gray, img_equal))    #numpy.hstack水平（按列）顺序stack数组
cv2.imshow("image", image)
cv2.waitKey()
plt.show()
'''


#彩色图均衡化
(b, g, r) = cv2.split(img)      #将三通道数组分开
b_equal = cv2.equalizeHist(b)   #对b通道进行均衡化
g_equal = cv2.equalizeHist(g)   #对g通道进行均衡化
r_equal = cv2.equalizeHist(r)   #对r通道进行均衡化

result = cv2.merge((b_equal, g_equal, r_equal))  #通道合成
img_result = np.hstack((img, result))
chans = cv2.split(result)
colors = ['b', 'g', 'r']
plt.figure()  #创建图像
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('彩色三通道-256')
plt.ylabel('数值')
plt.title('彩色图直方图')
for chan, color in zip(chans, colors):
    h = 0
    hist = cv2.calcHist([chan], [h], None, [256], [0, 256])   ##h处为传入图像的通道，灰度图只有一个通道0，彩色有三个通道0,1,2（bgr）
    #hist是一个shape为(256,1)的数组，表示0-255每个像素值对应的像素个数，下标即为相应的像素值
    ##plot一般需要输入x,y,若只输入一个参数，那么默认x为range(n)，n为y的长度
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
    h +=1
plt.show()
cv2.imshow("img result ", img_result)
cv2.waitKey()



