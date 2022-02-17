import cv2
import numpy as np

flag = True
#flag = False
if flag :
    img = cv2.imread("lenna.png") # 读入一张图片
    height,width,channels = img.shape
    #创建一张 上采样用的图片
    # uint8  无符号 0~255
    empty_image = np.zeros((700,700,channels),np.uint8)
    sh = height /700
    sw = width /700
    # 遍历放大后的图像，将对应位置的像素点与原图像对应起来
    for i in range(700):
        for j in range(700):
            x = int(i * sh)
            y = int(j * sw)
            empty_image[i,j] = img[x,y]
    cv2.imshow("img",img)
    cv2.imshow("empty_image",empty_image)
    cv2.waitKey(0)
#####################################

#flag = True
flag = False
if flag :
    img = cv2.imread("lenna.png") # 读入一张图片
    # 参数1 ：原图像
    # 参数2 ：输出图像，与比例因子二选其一
    # 参数3 ：沿水平轴的比例因子
    # 参数4 ：沿垂直轴的比例因子
    # 参数5 : 选择插值方式
    #res = cv2.resize(img,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_NEAREST)
    res = cv2.resize(img, (700,700) , interpolation=cv2.INTER_NEAREST)
    print(res.shape)
    cv2.imshow("img",img)
    cv2.imshow("res",res)
    cv2.waitKey(0)

## 知识点：
## int  作用相当于四舍五入
## uint8  无符号 
## 加了新的像素点，目标计算新的像素点的值，
## 通过最近邻值算法将原图对应的图像值赋值给新图