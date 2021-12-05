#灰度图


import cv2
import numpy as np


#1.灰度图  用opencv方法读取
img_cv = cv2.imread("lenna.png")  #cv读取图片，为numpy.ndarray格式
img_cv_h, img_cv_w, img_cv_c = img_cv.shape    #得到图片的宽高深信息   512,512,3
img_gray = np.zeros([img_cv_h, img_cv_w],  img_cv.dtype)

for i in range(img_cv_h):  #遍历每个像素点
    for j in range(img_cv_w):
        m_img = img_cv[i, j]  #取出原图像每个像素点的值
        img_gray[i, j] = int(m_img[0] * 0.11 + m_img[1] * 0.59 + m_img[2] * 0.3)  #将每个通道上的像素值乘以公式值用以转化为灰度图
        #print("img_gray:%s %s  %s",i,j, img_gray[i,j])
        #img_gray[]
print(img_gray.shape)
cv2.imshow("BRG - Gray", img_gray)
cv2.imshow("lenna", img_cv)
cv2.waitKey()

#img_gray = plt.imread("lenna.png")
#print(img_gray)

#二值图

print("进行二值图转化")

_, img_binary = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
print(img_binary)
cv2.imshow("binary", img_binary)
cv2.waitKey()





