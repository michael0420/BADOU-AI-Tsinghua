import cv2
import numpy as np

img = cv2.imread("lenna.png",1)
#将图像转化为灰度图
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 参数1 ： 灰度图像
# 参数2 ： 阈值1
# 参数3 ： 阈值2
res = cv2.Canny(img_gray,200,300)
cv2.imshow("img",img)
cv2.imshow("res",res)
cv2.waitKey(0)
print(img_gray.shape)


# Canny 算法的步骤
#1.将图像转换成灰度图，减少计算量
#2.对灰度图像进行高斯滤波，去除高频噪声
#3.检测图像中的边缘，使用sobel  算子比较好
#4.对图像进行非极大值抑制，进一步去噪声，选取局部最大值，非极大值点置为0 ，这样可以剔除一大部分非边缘的点
#5.用双阈值算法检测和连接边缘 ，进一步去噪声



