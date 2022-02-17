#coding:utf8

'''
直方图：
1.灰度直方图
2.彩色直方图
'''
import cv2
import matplotlib.pyplot as plt

'''
灰度直方图
1.plt.Hist
2.cv2.calcHist
'''


img=cv2.imread('lenna.png',1)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# # plt.Hist
# plt.figure()
# plt.hist(img_gray.ravel(),256)
# plt.show()

# #cv2.calcHist
# hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
# plt.figure()# 新建一个图像
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()

'''
彩色直方图【将通道分开计算】
'''
channels=cv2.split(img)
colors=("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (channel,color) in zip(channels,colors):
    hist=cv2.calcHist([channel],[0],None,[256],[0,256])
    print(type(hist))
    plt.plot(hist,color=color)
    plt.xlim([0,256])
plt.show()


