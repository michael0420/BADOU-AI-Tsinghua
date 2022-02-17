import matplotlib.pyplot as plt
import cv2
import numpy as np

#灰度化
def colourtogray(source_image):
    image=cv2.imread(source_image)                   #读入图像
    h,w=image.shape[:2]                              #获得图像高度和宽度值
    gray_image=np.zeros([h,w],image.dtype)           #新建同等大小图像
    for i in range(h):
        for j in range(w):
            b,g,r=image[i,j]                          #获得b,g,r三个通道的值
            gray_image[i,j]=int(b*0.11+g*0.59+r*0.3)  #灰度化计算（根据公式，灰度值=0.11b+0.59g+0.3r）
    print(gray_image)
    cv2.imshow("gray image",gray_image)               #显示图像
    cv2.waitKey(1000)                                 #延时10s(1000*10ms)
    return gray_image


#二值化
def graytowhiteblack(gray_image):
    h,w=gray_image.shape                              #获得图像高度和宽度值
    wbimage=np.zeros([h,w],gray_image.dtype)          #新建同等大小图像
    for i in range(h):
        for j in range(w):
            pixel=gray_image[i,j]/255.0               #将像素值归结到0，1范围内
            if pixel<=0.5:                            #二值化计算
                wbimage[i,j]=0
            else:
                wbimage[i,j]=255
    print(wbimage)
    cv2.imshow("white and black image",wbimage)       #显示图像
    cv2.waitKey(1000)                                 #延时10s(1000*10ms)
    return wbimage

new_img01=colourtogray("lenna.png")
new_img02=graytowhiteblack(new_img01)
cv2.imshow("gray image",new_img01)               #显示图像
cv2.waitKey(1000)
cv2.imshow("white and black image",new_img02)       #显示图像
cv2.waitKey(1000)

plt.subplot(221)
img_show1=plt.imread("lenna.png")
plt.imshow(img_show1)
print("---source image---")
print(img_show1)

plt.subplot(222)
img_show2=colourtogray("lenna.png")
plt.imshow(img_show2,cmap="gray")
print("---gray image---")
print(img_show2)

plt.subplot(223)
img_show3=graytowhiteblack(colourtogray("lenna.png"))
plt.imshow(img_show3,cmap="gray")
print("---white and black image---")
print(img_show3)
plt.show()
