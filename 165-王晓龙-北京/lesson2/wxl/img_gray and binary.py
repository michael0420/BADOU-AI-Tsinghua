from skimage.color import  rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import  cv2
############################################
flag = True
if flag :
    img =cv2.imread("lenna.png")
    # 获取图像的高和宽
    h,w = img.shape[:2]
    # 创建一张和原图像一样大小的图像
    img_gray = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            temp =img[i,j]
            # 注意opencv 读入的bgr  不时rgb
            # int 四舍五入 取整
            img_gray[i,j] = int(temp[0] * 0.11 + temp[1]*0.59 + temp[2]*0.3)

    img_binary =np.where(img_gray >=127,255,0)

    cv2.imshow("img",img) # 原图像
    cv2.imshow("img_gray",img_gray)  # 灰度化后的图像
    plt.imshow(img_binary,cmap="binary")
    plt.show()
    cv2.waitKey(0)

##################调用opencv###################
#flag =True
flag =False
if  flag :
    img = cv2.imread("lenna.png")
    # 将图像变成灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 参数1 灰度原图像
    # 参数2 阈值
    # 参数3 高于阈值时赋予的新值
    # 参数4 选择二值化参数
    ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # 返回值
    # ret  阈值
    # thresh  二值化图像
    # print(ret,thresh1)

    cv2.imshow("img", img)
    cv2.imshow("bin", thresh1)
    cv2.waitKey(0)

